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
def sample_inputs_cdist(op_info, device, dtype, requires_grad, **kwargs):
    small_S = 2
    test_cases = (((S, S, 2), (S, S + 1, 2)), ((S, S), (S, S)), ((S, S, S), (S, S, S)), ((3, 5), (3, 5)), ((2, 3, 5), (2, 3, 5)), ((1, 2, 3), (1, 2, 3)), ((1, 1), (S, 1)), ((0, 5), (4, 5)), ((4, 5), (0, 5)), ((0, 4, 5), (3, 5)), ((4, 5), (0, 3, 5)), ((0, 4, 5), (1, 3, 5)), ((1, 4, 5), (0, 3, 5)), ((small_S, small_S, small_S + 1, 2), (small_S, small_S, small_S + 2, 2)), ((small_S, 1, 1, small_S), (1, small_S, small_S)), ((1, 1, small_S), (small_S, 1, small_S, small_S)))
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
        for p in [0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, float('inf')]:
            for t1_size, t2_size in test_cases:
                yield SampleInput(make_arg(t1_size), make_arg(t2_size), p, cm)