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
def sample_inputs_linspace_tensor_overload(op, device, dtype, requires_grad, **kwargs):
    ends = (-3, 0, 1, 4, 50)
    starts = (-2.0, 0, 4.3, 50)
    nsteps = (0, 1, 50)
    is_start_end_tensors = ((True, True), (True, False), (False, True))
    make_arg = partial(torch.tensor, device=device, requires_grad=False)
    cases = list(product(starts, ends, nsteps, is_start_end_tensors)) + [(0, 7, 50, (True, True))]
    for start, end, nstep, (is_start_tensor, is_end_tensor) in cases:
        if dtype == torch.uint8 and (end < 0 or start < 0):
            continue
        tensor_options = {'dtype': dtype, 'device': device}
        if is_start_tensor:
            start = make_arg(start, dtype=torch.float32 if isinstance(start, float) else torch.int64)
        if is_end_tensor:
            end = make_arg(end, dtype=torch.float32 if isinstance(end, float) else torch.int64)
        yield SampleInput(start, args=(end, nstep), kwargs=tensor_options)
    yield SampleInput(1, args=(3, 1))