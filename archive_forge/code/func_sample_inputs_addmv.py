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
def sample_inputs_addmv(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases = (((S,), (S, M), (M,), 1, 1, False), ((S,), (S, M), (M,), 0.2, 0.6, False))
    test_cases_with_broadcast = (((1,), (S, M), (M,), 1, 1, True), ((1,), (S, M), (M,), 0.2, 0.6, True), ((), (S, M), (M,), 1, 1, True), ((), (S, M), (M,), 0.2, 0.6, True))
    cases = test_cases + test_cases_with_broadcast
    for size, mat, vec, beta, alpha, broadcasts_input in cases:
        yield SampleInput(make_arg(size), args=(make_arg(mat), make_arg(vec)), kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=broadcasts_input)