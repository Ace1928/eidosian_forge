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
def insert_activation_post_process(m, special_act_post_process=None):
    """ Adds an activation post process module and register
        a pre or post hook that calls the module
        """
    if needs_observation(m) and (not isinstance(m, DeQuantStub)):
        m.add_module('activation_post_process', get_activation_post_process(m.qconfig, device, special_act_post_process))
        _register_activation_post_process_hook(m, pre_hook=_activation_is_memoryless(m.qconfig))