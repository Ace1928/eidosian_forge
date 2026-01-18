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
def error_inputs_poisson_nll_loss(op_info, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5, 4),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='abc is not a valid value for reduction')
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5),)), error_regex='(Attempting to broadcast a dimension of length|The size of tensor a \\(5\\) must match the size of tensor b \\(4\\) at non-singleton dimension 1)')