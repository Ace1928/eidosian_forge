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
def module_inputs_torch_nn_MultiheadAttention(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = []
    bool_vals = (True, False)
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3, 3)))
    products = itertools.product(bool_vals, bool_vals, bool_vals, key_padding_masks, attn_masks)
    for bias, add_bias_kv, add_zero_attn, key_padding_mask, attn_mask in products:
        samples.append(ModuleInput(constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=True, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn), forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)), key_padding_mask=key_padding_mask, attn_mask=attn_mask), reference_fn=no_batch_dim_reference_mha))
        samples.append(ModuleInput(constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=False, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn), forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)), key_padding_mask=key_padding_mask, attn_mask=attn_mask), reference_fn=partial(no_batch_dim_reference_mha, batch_first=False)))
    return samples