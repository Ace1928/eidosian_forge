from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
def pack_params(self):
    w1 = []
    w2 = []
    for expert in self.experts:
        w1.append(expert.gate_up_proj.weight)
        w2.append(expert.down_proj.weight)
    self.w1 = torch._utils._flatten_dense_tensors(w1)
    w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)
    for data, param in zip(w1s, w1):
        param.data = data
    self.w1 = self.w1.view(len(w1), *w1s[0].shape)
    self.w2 = torch._utils._flatten_dense_tensors(w2)
    w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)
    for data, param in zip(w2s, w2):
        param.data = data
    self.w2 = self.w2.view(len(w2), *w2s[0].shape)