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
def reference_inputs_narrow_narrow_copy(op_info, device, dtype, requires_grad, *, is_narrow, **kwargs):
    yield from sample_inputs_narrow_narrow_copy(op_info, device, dtype, requires_grad, is_narrow=is_narrow, **kwargs)
    shapes_and_args = (((M,), 0, 0, 0), ((M,), -1, -1, 0), ((M,), 0, 5, 3), ((M,), 0, -5, 2), ((M,), -1, 0, M), ((M,), 0, -M, M), ((M, S), 1, 0, 0), ((S, M), -2, -1, 0), ((L, S), 1, 2, 3), ((L, S), -1, 3, 2), ((M, L), 0, 0, M), ((M, L), -1, -L, L), ((L, M, S), 2, 0, 0), ((M, S, L), -1, -1, 0), ((S, L, M), 2, 0, M), ((L, S, M), -1, -M, M), ((S, L, M), 1, 0, 0), ((S, L, M), 0, 2, 1), ((M, S, M), -1, -5, 4))
    for shape, dim, start, length in shapes_and_args:
        tensor = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(tensor, dim, start, length)
        if is_narrow:
            yield SampleInput(tensor, dim, torch.tensor(start), length)