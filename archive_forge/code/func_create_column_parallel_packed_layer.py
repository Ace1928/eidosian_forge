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
def create_column_parallel_packed_layer():
    if repeats == 2:
        linear = MergedColumnParallelLinear(4096, [4096] * repeats, bias=False)
        linear.weight.data = torch.rand_like(linear.weight.data)
        lora_linear = MergedColumnParallelLinearWithLoRA(linear)
    else:
        linear = QKVParallelLinear(4096, 64, 32, bias=False)
        linear.weight.data = torch.rand_like(linear.weight.data)
        lora_linear = QKVParallelLinearWithLora(linear)

    @dataclass
    class FakeConfig:
        hidden_size = 4096
        num_key_value_heads = 32
        num_attention_heads = 32
    lora_linear.create_lora_weights(max_loras, lora_config, model_config=FakeConfig())
    return (linear, lora_linear)