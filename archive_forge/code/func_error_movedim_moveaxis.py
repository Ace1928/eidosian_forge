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
def error_movedim_moveaxis(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((3, -3), (1, 0, -1))), error_regex='movedim: Invalid source or destination dims: source \\(\\[3, -3\\] dims\\) should contain the same number of dims as destination \\(\\[1, 0, -1\\] dims\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((3, -3, 4), (1, 0))), error_regex='movedim: Invalid source or destination dims: source \\(\\[3, -3, 4\\] dims\\) should contain the same number of dims as destination \\(\\[1, 0\\] dims\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((0, 4, -5), (1, 0, 2))), error_regex='movedim: repeated dim in `source` \\(\\[0, 4, -5\\]\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((1, 0, 2), (0, 4, -5))), error_regex='movedim: repeated dim in `destination` \\(\\[0, 4, -5\\]\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((1, 0, -4), (0, 4, -5))), error_regex='movedim: repeated dim in `source` \\(\\[1, 0, -4\\]\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((0, 1, -6), (1, 4, 2))), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((1, 4, 2), (0, 1, -6))), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=(-6, 1)), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=(3, -6)), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)