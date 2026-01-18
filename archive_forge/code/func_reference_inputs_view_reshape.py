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
def reference_inputs_view_reshape(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_view_reshape(op, device, dtype, requires_grad, **kwargs)
    cases = (((125,), (25, 5), True), ((25, 25), (1, 5, 5, 1, 5, 1, 5, 1), True), ((16, 32), (2, 4, 1, 4, 4, 1, 4), True), ((16, 12), (12, 16), True), ((1, 16, 12), (12, 16), True), ((1, 5, 1, 5), (25, 1), True), ((2, 4, 2), (4, 4), True), ((1, 4), (1, 1, 2, 1, 2), True), ((3, 5, 7), (7, 5, 3), True), ((1,), (), False), ((5, 0, 2, 3), (5, 0, 2, 3), True), ((2, 1, 0, 3, 1), (5, 0), True), ((1,), (), False), ((4, 5, 6), (4, 5, 6, 1, 1, 1), True), ((), (1, 1, 1, 1), False))
    irreversible_cases = (((), (-1,), False), ((4, 7, 9, 1, 1), (1, 4, 3, -1, 1), False))
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for a, b, is_tensor_supported in cases:
        if kwargs.get('tensor_arg') and (not is_tensor_supported):
            continue
        if kwargs.get('tensor_arg'):
            yield SampleInput(make_arg(a), args=(make_arg(b, requires_grad=False),))
            yield SampleInput(make_arg(b), args=(make_arg(a, requires_grad=False),))
        else:
            yield SampleInput(make_arg(a), args=(b,))
            yield SampleInput(make_arg(b), args=(a,))
    for a, b, is_tensor_supported in irreversible_cases:
        if kwargs.get('tensor_arg') and (not is_tensor_supported):
            continue
        if kwargs.get('tensor_arg'):
            b = make_arg(b, requires_grad=False)
        yield SampleInput(make_arg(a), args=(b,))