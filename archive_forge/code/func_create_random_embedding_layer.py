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