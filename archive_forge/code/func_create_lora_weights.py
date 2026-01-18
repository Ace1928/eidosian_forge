import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.punica import add_lora, add_lora_slice, bgmv
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import split_tensor_along_last_dim
def create_lora_weights(self, max_loras: int, lora_config: LoRAConfig, model_config: Optional[PretrainedConfig]=None) -> None:
    if 32000 < self.base_layer.vocab_size > 33024:
        raise ValueError('When using LoRA, vocab size must be 32000 >= vocab_size <= 33024')
    self.lora_a_stacked = torch.zeros((max_loras, 1, lora_config.max_lora_rank, self.hidden_size), dtype=lora_config.lora_dtype, device=self.device)
    self.lora_b_stacked = torch.zeros((max_loras, 1, math.ceil(self.base_layer.vocab_size / lora_config.lora_vocab_padding_size) * lora_config.lora_vocab_padding_size, lora_config.max_lora_rank), dtype=lora_config.lora_dtype, device=self.device)
    self.embeddings_tensors = torch.full((max_loras, lora_config.lora_extra_vocab_size, self.hidden_size), fill_value=float('-inf'), dtype=self.dtype, device=self.device)
    self.indices = None
    self.indices_padded = None
    self.indices_len = None