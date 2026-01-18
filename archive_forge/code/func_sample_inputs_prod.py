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
def sample_inputs_prod(op_info, device, dtype, requires_grad, **kwargs):

    def make_arg(shape):
        return make_tensor(shape, dtype=dtype, device=device, low=-1, high=+1, requires_grad=requires_grad)

    def prod_single_zero():
        result = make_arg(2 * (S,))
        result[0, 1] = 0
        return result
    for sample in sample_inputs_cumprod(op_info, device, dtype, requires_grad):
        yield SampleInput(sample.input.clone().requires_grad_(requires_grad))
        yield sample
    for sample in sample_inputs_cumprod(op_info, device, dtype, requires_grad):
        sample.kwargs['keepdim'] = True
        yield sample
    yield SampleInput(prod_single_zero())
    yield SampleInput(make_arg((3, 3, 3)), args=(1,))
    yield SampleInput(make_arg((3, 3, 3)), args=(1,), kwargs={'keepdim': True})
    yield SampleInput(make_arg((3, 0)), args=(1,))
    yield SampleInput(make_arg((3, 0)), args=(1,), kwargs={'keepdim': True})
    yield SampleInput(torch.tensor([2.0, 3, 0, 0], dtype=dtype, device=device, requires_grad=requires_grad))
    zero = make_arg(())
    zero.zero_()
    yield SampleInput(zero.clone().requires_grad_(requires_grad))
    yield SampleInput(zero.clone().requires_grad_(requires_grad), args=(0,))
    yield SampleInput(zero.clone().requires_grad_(requires_grad), args=(0,), kwargs={'keepdim': True})