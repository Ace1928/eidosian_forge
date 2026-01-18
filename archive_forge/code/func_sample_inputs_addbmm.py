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
def sample_inputs_addbmm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = [((S, M), (S, S, S), (S, S, M), 1, 1, False), ((1,), (S, S, S), (S, S, M), 1, 1, True), ((S, M), (S, S, S), (S, S, M), 0.6, 0.2, False), ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True), ((), (S, S, S), (S, S, M), 1, 1, True), ((), (S, S, S), (S, S, M), 0.6, 0.2, True)]
    for input_shape, batch1_shape, batch2_shape, beta, alpha, is_broadcasting in test_cases:
        if dtype.is_complex:
            beta_complex, alpha_complex = (beta * (1 + 2j), alpha * (2 + 3j))
            yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)), kwargs=dict(beta=beta_complex, alpha=alpha_complex), broadcasts_input=is_broadcasting)
        yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)), kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=is_broadcasting)