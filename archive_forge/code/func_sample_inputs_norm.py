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
def sample_inputs_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = [((S, S), (2,), '2'), ((S, S), (0,), '0'), ((S, S), (0.5,), '0_5'), ((S, S), (1,), '1'), ((S, S), (3,), '3'), ((S, S), (-1,), 'neg_1'), ((S, S), (-2,), 'neg_2'), ((S, S), (-0.5,), 'neg_0_5'), ((S, S), (-1.5,), 'neg_1_5')]
    cases_nonzero_input = (((S, S, S), (1.5,), '1_5_default'), ((S, S, S), (1.5, 1), '1_5_dim'), ((S, S, S), (1.5, -1), '1_5_neg_dim'), ((S, S, S), (1.5, 1, True), 'keepdim_1_5_dim'), ((S, S, S), (1.5, -1, True), 'keepdim_1_5_neg_dim'))
    cases_posdim = (((S, S), (-2, 1), 'neg_2_dim'), ((S, S), (-1, 1), 'neg_1_dim'), ((S, S), (0, 1), '0_dim'), ((S, S), (1, 1), '1_dim'), ((S, S), (2, 1), '2_dim'), ((S, S), (3, 1), '3_dim'), ((S, S, S), (2, 1), '2_dim'), ((S, S, S), (3, 1), '3_dim'), ((S, S, S), (2, 1, True), 'keepdim_2_dim'), ((S, S, S), (3, 1, True), 'keepdim_3_dim'), ((), (2, 0), '2_dim_scalar'), ((), (3, 0), '3_dim_scalar'), ((), (2, 0, True), 'keepdim_2_dim_scalar'), ((), (3, 0, True), 'keepdim_3_dim_scalar'))
    cases_negdim = ((shape, args[:1] + (-args[1],) + args[2:], name.replace('_dim', '_neg_dim')) for shape, args, name in cases_posdim)
    for shape, args, name in itertools.chain(cases, cases_posdim, cases_negdim):
        yield SampleInput(make_arg(shape), args=args, name=name)
    for shape, args, name in cases_nonzero_input:
        yield SampleInput(make_arg(shape, exclude_zero=True), args=args, name=name)