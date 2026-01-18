import warnings
from typing import List, Optional
import bitsandbytes as bnb
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .layer import LoraLayer
def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    loaded_in_4bit = kwargs.get('loaded_in_4bit', False)
    if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        fourbit_kwargs = kwargs.copy()
        fourbit_kwargs.update({'compute_dtype': target_base_layer.compute_dtype, 'compress_statistics': target_base_layer.weight.compress_statistics, 'quant_type': target_base_layer.weight.quant_type})
        new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)
    return new_module