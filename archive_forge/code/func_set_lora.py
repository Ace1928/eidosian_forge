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
def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor, embeddings_tensor: Optional[torch.Tensor]):
    self.reset_lora(index)
    self.lora_a_stacked[index, 0, :lora_a.shape[1], :lora_a.shape[0]].copy_(lora_a.T, non_blocking=True)
    self.lora_b_stacked[index, 0, :lora_b.shape[1], :lora_b.shape[0]].copy_(lora_b.T, non_blocking=True)
    if embeddings_tensor is not None:
        self.embeddings_tensors[index, :embeddings_tensor.shape[0], :embeddings_tensor.shape[1]] = embeddings_tensor