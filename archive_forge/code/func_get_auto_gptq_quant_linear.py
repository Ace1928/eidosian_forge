import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple
import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size
from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
def get_auto_gptq_quant_linear(gptq_quantization_config):
    """
    Get the right AutoGPTQQuantLinear class based on the quantization config file
    """
    if gptq_quantization_config is not None and is_auto_gptq_available():
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
        desc_act = gptq_quantization_config.desc_act
        group_size = gptq_quantization_config.group_size
        bits = gptq_quantization_config.bits
        if hasattr(gptq_quantization_config, 'use_exllama'):
            use_exllama = gptq_quantization_config.use_exllama
        else:
            use_exllama = not gptq_quantization_config.disable_exllama
        if hasattr(gptq_quantization_config, 'exllama_config'):
            exllama_version = gptq_quantization_config.exllama_config['version']
        else:
            exllama_version = 1
        AutoGPTQQuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=not (use_exllama and exllama_version == 1), disable_exllamav2=not (use_exllama and exllama_version == 2))
        return AutoGPTQQuantLinear
    return None