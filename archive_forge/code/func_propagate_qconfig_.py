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
def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None):
    """Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)
        prepare_custom_config_dict: dictionary for custom handling of modules
            see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    _propagate_qconfig_helper(module, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)