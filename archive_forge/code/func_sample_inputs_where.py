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
def sample_inputs_where(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def make_bool_mask(shape):
        mask_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)
        if mask_t.numel() == 0:
            return mask_t
        elif mask_t.numel() == 1:
            mask_t.fill_(True)
            return mask_t
        if mask_t.sum() == 0:

            def random_index(shape):
                return tuple((random.randrange(0, max_idx) for max_idx in shape))
            mask_t[random_index(mask_t.shape)] = True
            return mask_t
        return mask_t
    cases = (((M, M), (M, M), (M, M), False), ((M, 1, M), (M, M), (M, M, 1), True), ((), (), (), False), ((M, 1, M), (), (M, M, 1), True), ((), (M, M), (), True), ((), 2, (1, 1), True))
    for shape, mask_shape, other_shape, broadcasts_input in cases:
        yield SampleInput(make_arg(shape), args=(make_bool_mask(mask_shape), make_arg(other_shape)), broadcasts_input=broadcasts_input)