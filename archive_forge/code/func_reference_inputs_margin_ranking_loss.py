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
def reference_inputs_margin_ranking_loss(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_margin_ranking_loss(op, device, dtype, requires_grad, **kwargs)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for reduction in ('sum', 'mean', 'none'):
        if dtype.is_floating_point:
            inp1 = make_input((10,))
            inp1[2] = float('nan')
            inp2 = make_input((10,))
            inp2[4] = float('nan')
            target = make_input((10,))
            inp2[9] = float('nan')
            yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})
            inp1 = make_input((10,))
            inp2[1] = float('inf')
            inp2 = make_input((10,))
            inp2[4] = float('inf')
            target = make_input((10,))
            inp2[7] = float('inf')
            yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})
        inp1 = make_input((5, 2))
        inp2 = make_input((5, 1))
        target = make_input((1, 2))
        yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})