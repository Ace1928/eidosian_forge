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
def reference_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs)
    supported_dtypes = op_info.supported_dtypes(device)
    make_arg = partial(make_tensor, device=device, requires_grad=requires_grad)
    types = ((torch.float64, torch.complex128), (torch.bfloat16, torch.float32))
    values = (None, True, False, 3.14, 3, 1.0, 1, 0.0, 0, -3.14, -3, 3.14 + 2.71j)
    for (type2, type3), value in product(types, values):
        if type2 not in supported_dtypes or type3 not in supported_dtypes:
            continue
        if type(value) is complex and type2 is not torch.complex128:
            continue
        arg1 = make_arg([5, 5], dtype=dtype)
        arg2 = make_arg([5, 5], dtype=type2)
        arg3 = make_arg([1, 5], dtype=type3)
        if value is not None:
            yield SampleInput(arg1, args=(arg2, arg3), kwargs=dict(value=value))
        else:
            yield SampleInput(arg1, args=(arg2, arg3))