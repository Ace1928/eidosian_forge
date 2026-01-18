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
def error_inputs_avg_pool2d(op_info, device, **kwargs):
    x = torch.rand([0, 1, 49], dtype=torch.float32)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': -1}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2), 'stride': 50, 'padding': -1}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 4}), error_regex='pad should be at most half of kernel size')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2), 'stride': 50, 'padding': 4}), error_regex='pad should be at most half of kernel size')
    x = torch.zeros(3, 3, 3)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (2, 2), 'divisor_override': 0}), error_regex='divisor must be not zero')