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
def reference_inputs_diagonal_diag_embed(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_diagonal_diag_embed(op_info, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes1d = ((0,), (1,))
    shapes2d = ((L, M),)
    shapes3d = ((L, M, S),)
    kwargs1d = {}
    kwargs2d = (dict(dim1=1, dim2=0), dict(dim1=-2, dim2=-1), dict(offset=100))
    kwargs3d = kwargs2d + (dict(offset=-1, dim1=0, dim2=2),)
    samples1d = product(shapes1d, kwargs1d)
    samples2d = product(shapes2d, kwargs2d)
    samples3d = product(shapes3d, kwargs3d)
    for shape, kwargs in chain(samples1d, samples2d, samples3d):
        if 'diagonal' in op_info.name:
            if shape in ((0,), (1,)):
                continue
        yield SampleInput(input=make_arg(shape), kwargs=kwargs)