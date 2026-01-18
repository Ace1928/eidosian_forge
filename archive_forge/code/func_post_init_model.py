import json
import os
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod
from ..utils import is_accelerate_available, is_auto_gptq_available
from ..utils.modeling_utils import recurse_getattr
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import get_block_name_with_pattern, get_device, get_layers, get_preceding_modules, get_seqlen
def post_init_model(self, model):
    """
        Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """
    if self.bits == 4 and (not self.disable_exllama):
        if get_device(model) == torch.device('cpu') or (hasattr(model, 'hf_device_map') and any((d in model.hf_device_map for d in ['cpu', 'disk']))):
            raise ValueError('Found modules on cpu/disk. Using Exllama or Exllamav2 backend requires all the modules to be on GPU.You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object')

    class StoreAttr(object):
        pass
    model.quantize_config = StoreAttr()
    model.quantize_config.desc_act = self.desc_act
    model = autogptq_post_init(model, use_act_order=self.desc_act)
    if self.desc_act and (not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE) and (self.max_input_length is not None):
        model = exllama_set_max_input_length(model, self.max_input_length)
    return model