from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def get_total_num_kv_heads(self) -> int:
    """Returns the total number of KV heads."""
    falcon_model_types = ['falcon', 'RefinedWeb', 'RefinedWebModel']
    new_decoder_arch_falcon = self.hf_config.model_type in falcon_model_types and getattr(self.hf_config, 'new_decoder_architecture', False)
    if not new_decoder_arch_falcon and getattr(self.hf_config, 'multi_query', False):
        return 1
    attributes = ['n_head_kv', 'num_kv_heads', 'num_key_value_heads', 'multi_query_group_num']
    for attr in attributes:
        num_kv_heads = getattr(self.hf_config, attr, None)
        if num_kv_heads is not None:
            return num_kv_heads
    return self.hf_config.num_attention_heads