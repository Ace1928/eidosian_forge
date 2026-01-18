import copy
import itertools
import warnings
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization.quantization_mappings import (
from .utils import get_qparam_dict, has_no_children_ignoring_parametrizations
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
from torch.ao.quantization.qconfig import (
from torch.nn.utils.parametrize import type_before_parametrizations
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.observer import (   # noqa: F401
def add_quant_dequant(module):
    """Wrap the leaf child module in QuantWrapper if it has a valid qconfig
    Note that this function will modify the children of module inplace and it
    can return a new module which wraps the input module as well.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize

    Return:
        Either the inplace modified module with submodules wrapped in
        `QuantWrapper` based on qconfig or a new `QuantWrapper` module which
        wraps the input module, the latter case only happens when the input
        module is a leaf module and we want to quantize it.
    """
    if has_no_children_ignoring_parametrizations(module) and hasattr(module, 'qconfig') and module.qconfig:
        return QuantWrapper(module)
    for name, child in module.named_children():
        module._modules[name] = add_quant_dequant(child)
    return module