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
def reference_group_norm(inp: np.ndarray, num_groups: int, weight=None, bias=None, eps=1e-05):
    inp_view = inp
    if np.prod(inp.shape) != 0:
        inp_view = inp.reshape((inp.shape[0], num_groups, -1))
    mean = inp_view.mean(axis=-1, keepdims=True)
    var = inp_view.var(axis=-1, ddof=0, keepdims=True)
    Y = (inp_view - mean) / np.sqrt(var + eps)
    Y = Y.reshape(inp.shape)
    if weight is not None:
        if len(Y.shape) > 2:
            weight = np.expand_dims(weight, [0] + [idx + 2 for idx in range(inp.ndim - 2)])
        Y = Y * weight
    if bias is not None:
        if len(Y.shape) > 2:
            bias = np.expand_dims(bias, [0] + [idx + 2 for idx in range(inp.ndim - 2)])
        Y = Y + bias
    return Y