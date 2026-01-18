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
def sample_inputs_reduction_quantile(op_info, device, dtype, requires_grad, **kwargs):
    test_quantiles = (0.5, make_tensor((2,), dtype=dtype, device=device, low=0, high=1, requires_grad=requires_grad))
    test_interpolations = ['linear', 'midpoint']
    for quantiles in test_quantiles:
        for t in _generate_reduction_inputs(device, dtype, requires_grad):
            input = t.clone().requires_grad_(requires_grad)
            yield SampleInput(input, quantiles)
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims=False):
                kwargs.setdefault('dim', 0)
                kwargs.setdefault('keepdim', False)
                for interpolation in test_interpolations:
                    kwargs['interpolation'] = interpolation
                    input = t.clone().requires_grad_(requires_grad)
                    yield SampleInput(input, quantiles, **kwargs)