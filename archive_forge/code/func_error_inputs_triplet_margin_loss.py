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
def error_inputs_triplet_margin_loss(op_info, device, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    samples = ((make_input(3, 4), (make_input(3, 4), make_input(3, 4)), dict(reduction='abc'), ValueError, 'abc is not a valid value for reduction'), (make_input(3, 5), (make_input(3, 4), make_input(3, 4)), dict(), RuntimeError, '(Attempting to broadcast a dimension of length|The size of tensor a \\(5\\) must match the size of tensor b \\(4\\) at non-singleton dimension 1)'), (make_input(3, 4), (make_input(3, 5), make_input(3, 4)), dict(), RuntimeError, '(Attempting to broadcast a dimension of length|The size of tensor a \\(4\\) must match the size of tensor b \\(5\\) at non-singleton dimension 1)'), (make_input(3, 4), (make_input(3, 4), make_input(3, 5)), dict(), RuntimeError, '(Attempting to broadcast a dimension of length|The size of tensor a \\(4\\) must match the size of tensor b \\(5\\) at non-singleton dimension 1)'), (make_input(3), (make_input(3, 4), make_input(3, 4)), dict(), RuntimeError, 'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor 1D, positive 2D, and negative 2D inputs'), (make_input(3, 4), (make_input(3), make_input(3, 4)), dict(), RuntimeError, 'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor 2D, positive 1D, and negative 2D inputs'), (make_input(3, 4), (make_input(3, 4), make_input(3)), dict(), RuntimeError, 'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor 2D, positive 2D, and negative 1D inputs'))
    for input, args, kwargs, error_type, error_regex in samples:
        yield ErrorInput(SampleInput(input, args=args, kwargs=kwargs), error_type=error_type, error_regex=error_regex)