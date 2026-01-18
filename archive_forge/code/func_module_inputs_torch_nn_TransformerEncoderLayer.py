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
def module_inputs_torch_nn_TransformerEncoderLayer(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [ModuleInput(constructor_input=FunctionInput(4, 2, 16, 0.0), forward_input=FunctionInput(make_input((2, 3, 4))), desc='relu_activation'), ModuleInput(constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu), forward_input=FunctionInput(make_input((2, 3, 4))), desc='gelu_activation')]
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for src_mask, src_key_padding_mask, norm_first in itertools.product(attn_masks, key_padding_masks, (True, False)):
        samples.append(ModuleInput(constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8, dropout=0.0, batch_first=True, norm_first=norm_first), forward_input=FunctionInput(make_input((3, 4)), src_mask=src_mask, src_key_padding_mask=src_key_padding_mask), reference_fn=partial(no_batch_dim_reference_fn, batch_first=True, kwargs_to_batchify={'src_key_padding_mask': 0}), desc='no_batch_dim_batch_first'))
        samples.append(ModuleInput(constructor_input=FunctionInput(4, 2, 8, dropout=0.0, batch_first=False, norm_first=norm_first), forward_input=FunctionInput(make_input((3, 4)), src_mask=src_mask, src_key_padding_mask=src_key_padding_mask), reference_fn=partial(no_batch_dim_reference_fn, batch_first=False, kwargs_to_batchify={'src_key_padding_mask': 0}), desc='no_batch_dim'))

    def fast_path_reference_fn(module, parameters, *args, **kwargs):
        assert not module.training
        module = module.train(True)
        output = module(*args, **kwargs)
        module = module.train(False)
        return output
    if not training:
        for norm_first in (True, False):
            samples.append(ModuleInput(constructor_input=FunctionInput(4, 2, 8, dropout=0.0, batch_first=True, norm_first=norm_first), forward_input=FunctionInput(make_input((2, 3, 4))), reference_fn=fast_path_reference_fn, desc='fast_path_norm_first' if norm_first else 'fast_path'))
    return samples