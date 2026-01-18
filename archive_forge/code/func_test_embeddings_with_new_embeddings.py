import pytest
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn.functional as F
from vllm.lora.layers import (
from vllm.lora.models import LoRALayerWeights, convert_mapping, PackedLoRALayerWeights
from vllm.config import LoRAConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.utils import set_random_seed
from .utils import DummyLoRAManager
@torch.inference_mode()
@pytest.mark.parametrize('num_loras', [1, 2, 4, 8])
@pytest.mark.parametrize('device', CUDA_DEVICES)
def test_embeddings_with_new_embeddings(dist_init, num_loras, device) -> None:
    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras, max_lora_rank=8, lora_dtype=torch.float16)

    def create_random_embedding_layer():
        embedding = VocabParallelEmbedding(512, 256)
        embedding_data = torch.rand_like(embedding.weight.data)
        embedding.weight.data = embedding_data
        embedding.weight.data[512:, :] = 0
        expanded_embedding = VocabParallelEmbedding(512 + lora_config.lora_extra_vocab_size * max_loras, 256, org_num_embeddings=512)
        expanded_embedding.weight.data[:512, :] = embedding_data
        lora_embedding = VocabParallelEmbeddingWithLoRA(deepcopy(expanded_embedding))
        lora_embedding.create_lora_weights(max_loras, lora_config)
        return (expanded_embedding, lora_embedding)
    for i in range(10):
        set_random_seed(i)
        id_to_index = get_random_id_to_index(num_loras, max_loras)
        expanded_embedding, lora_embedding = create_random_embedding_layer()
        lora_dict, _ = populate_loras(id_to_index, layer=lora_embedding, layer_weights=torch.zeros((256, 512 + lora_config.lora_extra_vocab_size)), generate_embeddings_tensor=256)
        embeddings_tensors = [lora_dict[id].embeddings_tensor for id in sorted(lora_dict.keys())]
        embeddings_tensor_len = embeddings_tensors[0].shape[0]
        for _ in range(max_loras - len(embeddings_tensors)):
            embeddings_tensors.append(torch.zeros(embeddings_tensors[0].shape))
        inputs, index_mapping, prompt_mapping = create_random_inputs(active_lora_ids=list(lora_dict.keys()), num_inputs=num_loras * 3, input_size=(200,), input_range=(1, 512))
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        original_inputs = deepcopy(inputs)
        for input_, original_input_, lora_id in zip(inputs, original_inputs, prompt_mapping):
            embedding_id = lora_id - 1
            input_[-1] = 512 + embedding_id * embeddings_tensor_len
            original_input_[-1] = 512
            input_[-2] = 512 + ((embedding_id + 1) * embeddings_tensor_len - 1)
            original_input_[-2] = 512 + embeddings_tensor_len - 1
        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras, 512, lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info)
        expanded_embedding.weight[512:512 + embeddings_tensor_len * max_loras] = torch.cat(embeddings_tensors)
        lora_result = lora_embedding(torch.cat(original_inputs))
        expected_results = []
        for input_, original_input_, lora_id in zip(inputs, original_inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = expanded_embedding(input_)
            after_a = F.embedding(original_input_, lora.lora_a)
            result += after_a @ lora.lora_b
            expected_results.append(result)
        expected_result = torch.cat(expected_results)
        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result, expected_result, rtol=rtol, atol=atol)
        for slot_idx in range(max_loras):
            lora_embedding.reset_lora(slot_idx)
        inputs, index_mapping, prompt_mapping = create_random_inputs(active_lora_ids=[0], num_inputs=num_loras * 3, input_size=(200,), input_range=(1, 512))
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        original_inputs = deepcopy(inputs)
        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras, 512, lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info)
        lora_result = lora_embedding(torch.cat(original_inputs))
        expected_result = expanded_embedding(torch.cat(inputs))
        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result, expected_result, rtol=rtol, atol=atol)