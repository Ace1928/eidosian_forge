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
def populate_loras(id_to_index: List[Optional[int]], layer: BaseLayerWithLoRA, layer_weights: torch.Tensor, generate_embeddings_tensor: int=0, repeats: int=1) -> Tuple[Dict[int, LoRALayerWeights], Dict[int, List[LoRALayerWeights]]]:
    """This method populates the lora layers with lora weights.

    Args:
        id_to_index: a list of lora ids. The index of the lora id
            represents which memory slot the lora matrices are
            stored in. A None value indicates a free slot.
        layer: the LoRAlayer to populate.
        layer_weights: the PyTorch tensor containing the layer's
            weights.
        generate_embeddings_tensor: whether to generate an
            embeddings tensor for each LoRA.
        repeats: must only be set for column parallel packed
            layers. Indicates the number of loras to compose
            together to create a single lora layer.
    """
    lora_dict: Dict[int, LoRALayerWeights] = dict()
    sublora_dict: Dict[int, List[LoRALayerWeights]] = dict()
    for slot_idx, lora_id in enumerate(id_to_index):
        if lora_id is not None:
            subloras = []
            sublora_len = layer_weights.shape[0] // repeats
            for i in range(repeats):
                sublora = DummyLoRAManager().init_random_lora(module_name=f'fake_{i}', weight=layer_weights, generate_embeddings_tensor=generate_embeddings_tensor)
                sublora.lora_b = sublora.lora_b[:, sublora_len * i:sublora_len * (i + 1)]
                sublora.optimize()
                subloras.append(sublora)
            lora = PackedLoRALayerWeights.pack(subloras) if repeats > 1 else subloras[0]
            layer.set_lora(slot_idx, lora_a=lora.lora_a, lora_b=lora.lora_b, embeddings_tensor=lora.embeddings_tensor)
            lora_dict[lora_id] = lora
            sublora_dict[lora_id] = subloras
    return (lora_dict, sublora_dict)