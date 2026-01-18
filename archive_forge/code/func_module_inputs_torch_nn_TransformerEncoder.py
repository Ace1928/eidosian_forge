import torch
import unittest
from copy import deepcopy
from enum import Enum
from functools import wraps, partial
from itertools import chain, product
import itertools
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_nn import nllloss_reference, get_reduction
from torch.testing._internal.common_utils import (
from types import ModuleType
from typing import List, Tuple, Type, Set, Dict
def module_inputs_torch_nn_TransformerEncoder(module_info, device, dtype, requires_grad, training, **kwargs):
    samples = []
    for layer_module_input in module_inputs_torch_nn_TransformerEncoderLayer(None, device, dtype, requires_grad, training):
        l_args, l_kwargs = (layer_module_input.constructor_input.args, layer_module_input.constructor_input.kwargs)
        l_kwargs['device'] = device
        l_kwargs['dtype'] = dtype
        encoder_layer = torch.nn.TransformerEncoderLayer(*l_args, **l_kwargs)
        num_layers = 2
        forward_input = layer_module_input.forward_input
        if 'src_mask' in forward_input.kwargs:
            forward_input.kwargs['mask'] = forward_input.kwargs['src_mask']
            del forward_input.kwargs['src_mask']
        samples.append(ModuleInput(constructor_input=FunctionInput(encoder_layer, num_layers), forward_input=forward_input, desc=layer_module_input.desc))
    return samples