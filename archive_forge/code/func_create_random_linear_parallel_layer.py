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
def create_random_linear_parallel_layer():
    if orientation == 'row':
        linear = RowParallelLinear(4096, 4096, bias=False)
        linear.weight.data = torch.rand_like(linear.weight.data)
        lora_linear = RowParallelLinearWithLoRA(linear)
    else:
        linear = ColumnParallelLinear(4096, 4096, bias=False)
        linear.weight.data = torch.rand_like(linear.weight.data)
        lora_linear = ColumnParallelLinearWithLoRA(linear)
    lora_linear.create_lora_weights(max_loras, lora_config)
    return (linear, lora_linear)