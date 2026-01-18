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
def sample_inputs_ormqr(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, dtype=dtype, device=device, low=-1, high=1)
    batches = [(), (0,), (2,), (2, 1)]
    ns = [5, 2, 0]
    tf = [True, False]
    for batch, (m, n), left, transpose in product(batches, product(ns, ns), tf, tf):
        input = make_input((*batch, m, n))
        reflectors, tau = torch.geqrf(input)
        reflectors.requires_grad_(requires_grad)
        tau.requires_grad_(requires_grad)
        other_matrix_shape = (m, n) if left else (n, m)
        other = make_input((*batch, *other_matrix_shape), requires_grad=requires_grad)
        yield SampleInput(reflectors, tau, other, left=left, transpose=transpose)