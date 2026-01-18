import warnings
from typing import List, Optional
import bitsandbytes as bnb
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .layer import LoraLayer
def dispatch_bnb_8bit(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    loaded_in_8bit = kwargs.get('loaded_in_8bit', False)
    if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
        eightbit_kwargs = kwargs.copy()
        eightbit_kwargs.update({'has_fp16_weights': target.state.has_fp16_weights, 'memory_efficient_backward': target.state.memory_efficient_backward, 'threshold': target.state.threshold, 'index': target.index})
        new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)
    return new_module