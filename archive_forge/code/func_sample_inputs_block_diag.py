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
def sample_inputs_block_diag(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases: Tuple[tuple] = (((1, S), (2, S), (3, S)), ((S, 1), (S, 2), (S, 3)), ((1,), (2,), (3,)), ((2, S), (S,)))
    for shape, *other_shapes in test_cases:
        yield SampleInput(make_arg(shape), args=tuple((make_arg(s) for s in other_shapes)))
        if dtype == torch.complex32 or dtype == torch.complex64:
            non_complex_dtype = torch.float32 if dtype == torch.complex32 else torch.float64
            make_arg_non_complex = partial(make_tensor, dtype=non_complex_dtype, device=device, requires_grad=requires_grad)
            yield SampleInput(make_arg_non_complex(shape), args=tuple((make_arg(s) for s in other_shapes)))