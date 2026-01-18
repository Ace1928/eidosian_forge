from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def sample_inputs_max_unpool(op_info, device, dtype, requires_grad, **kwargs):
    unpool_name_to_pool_method_dict = {'nn.functional.max_unpool1d': torch.nn.functional.max_pool1d, 'nn.functional.max_unpool2d': torch.nn.functional.max_pool2d, 'nn.functional.max_unpool3d': torch.nn.functional.max_pool3d}
    unpool_name_to_dim = {'nn.functional.max_unpool1d': 1, 'nn.functional.max_unpool2d': 2, 'nn.functional.max_unpool3d': 3}
    unpool_to_pool_name_dict = {k: f'nn.functional.{v.__name__}' for k, v in unpool_name_to_pool_method_dict.items()}
    pool_dim = unpool_name_to_dim[op_info.name]
    pool_method = unpool_name_to_pool_method_dict[op_info.name]
    pool_op_info = copy.copy(op_info)
    pool_op_info.name = unpool_to_pool_name_dict[op_info.name]
    for sample in sample_inputs_max_pool(pool_op_info, device, dtype, requires_grad, **kwargs):
        if sample.input.dim() != pool_dim + 2:
            continue
        if sample.kwargs['dilation'] != 1:
            continue
        if sample.kwargs['return_indices']:
            pool, indices = pool_method(sample.input, **sample.kwargs)
            arg = pool.detach().requires_grad_(requires_grad)
            sample_kwargs = {'kernel_size': sample.kwargs['kernel_size'], 'stride': sample.kwargs['stride'], 'padding': sample.kwargs['padding'], 'output_size': sample.input.size()}
            yield SampleInput(arg, args=(indices,), kwargs=sample_kwargs)