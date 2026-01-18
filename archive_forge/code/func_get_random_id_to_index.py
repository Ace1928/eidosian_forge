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
def get_random_id_to_index(num_loras: int, num_slots: int, log: bool=True) -> List[Optional[int]]:
    """Creates a random lora_id_to_index mapping.

    Args:
        num_loras: The number of active loras in the mapping.
        num_slots: The number of slots in the mapping. Must be larger
            than num_loras.
        log: Whether to log the output.
    """
    if num_loras > num_slots:
        raise ValueError(f'num_loras is higher than num_slots: {num_loras} > {num_slots}. num_loras must be less than or equal to num_slots.')
    slots: List[Optional[int]] = [None] * num_slots
    random_slot_selections = torch.randperm(num_slots)[:num_loras].tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        slots[slot_idx] = lora_id
    if log:
        print(f'Created lora_id_to_index mapping: {slots}.')
    return slots