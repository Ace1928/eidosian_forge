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
def error_inputs_dot_vdot(op_info, device, is_ref=False, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    if not is_ref:
        yield ErrorInput(SampleInput(make_input(1), args=(make_input(3, dtype=torch.float16),)), error_regex='dot : expected both vectors to have same dtype')
    yield ErrorInput(SampleInput(make_input(1, 1), args=(make_input(3),)), error_regex='1D tensors expected')
    yield ErrorInput(SampleInput(make_input(9), args=(make_input(3),)), error_regex='inconsistent tensor size')
    if device != 'cpu' and (not is_ref):
        yield ErrorInput(SampleInput(make_input(3), args=(make_input(3, device='cpu'),)), error_regex='Expected all tensors to be on the same device')