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
def reference_inputs_logsumexp(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_logsumexp(op, device, dtype, requires_grad, **kwargs)
    t = torch.tensor([20, 30, 100], dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t, 0, False)
    t = torch.tensor((), dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t, 0, False)
    t = torch.tensor(float('inf'))
    yield SampleInput(t, 0, True)