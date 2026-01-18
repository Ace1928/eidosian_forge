from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def get_head_size(self) -> int:
    if hasattr(self.hf_config, 'head_dim'):
        return self.hf_config.head_dim
    return self.hf_config.hidden_size // self.hf_config.num_attention_heads