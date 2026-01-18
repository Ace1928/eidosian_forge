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
def sample_inputs_add_sub(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)
    if dtype is not torch.bool:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': 2})
    else:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': True})
    neg_alpha = -3.125 if dtype.is_floating_point or dtype.is_complex else -3
    lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)
    if dtype is not torch.bool:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': neg_alpha})
    else:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': False})