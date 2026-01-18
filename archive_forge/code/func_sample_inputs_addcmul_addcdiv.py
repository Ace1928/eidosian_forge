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
def sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = [(((S, S), (S, S), (S, S)), False), (((S, S), (S, 1), (1, S)), False), (((1,), (S, S, 1), (1, S)), True), (((), (), ()), False), (((S, S), (), ()), True), (((), (S, S, 1), (1, S)), True)]
    for input_args, broadcasts_input in test_cases:
        args = tuple((make_arg(arg, exclude_zero=True) if isinstance(arg, tuple) else arg for arg in input_args))
        yield SampleInput(*args).with_metadata(broadcasts_input=broadcasts_input)
        args = tuple((make_arg(arg, exclude_zero=True) if isinstance(arg, tuple) else arg for arg in input_args))
        yield SampleInput(*args, value=3.14 if dtype.is_floating_point or dtype.is_complex else 3).with_metadata(broadcasts_input=broadcasts_input)