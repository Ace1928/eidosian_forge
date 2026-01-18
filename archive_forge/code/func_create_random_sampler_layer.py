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
def create_random_sampler_layer():
    linear = ParallelLMHead(32000 + lora_config.lora_extra_vocab_size, 1024, 32000)
    linear.weight.data = torch.rand_like(linear.weight.data)
    linear.weight.data[:, 32000:] = 0
    sampler = Sampler(32000 + lora_config.lora_extra_vocab_size, 32000)
    lora_sampler = SamplerWithLoRA(sampler, 1024, linear.weight.dtype, linear.weight.device)
    lora_sampler.create_lora_weights(max_loras, lora_config)
    return (linear, sampler, lora_sampler)