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
def sample_inputs_lerp(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)), make_arg((S, S)), 0.4)
    yield SampleInput(make_arg((S, S)), make_arg((S,)), 0.4)
    yield SampleInput(make_arg(()), make_arg(()), 0.4)
    yield SampleInput(make_arg((S, S)), make_arg(()), 0.4)
    yield SampleInput(make_arg((S, S)), make_arg((S,)), make_arg((S, S)))
    yield SampleInput(make_arg((S, S)), make_arg((S, 1)), make_arg((S,)))
    yield SampleInput(make_arg((S,)), make_arg((S, S)), 0.4).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg(()), make_arg((S, S)), 0.4).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, 1)), make_arg((S, S)), 0.4).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, 1)), make_arg((S, S)), make_arg((S, 1))).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, S)), make_arg((S, S)), make_arg((S, S)))
    yield SampleInput(make_arg((S,)), make_arg((S, S)), make_arg((S, S))).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S,)), make_arg((S, S, S)), make_arg((S, S))).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, S)), make_arg((S, S, S)), make_arg((S,))).with_metadata(broadcasts_input=True)
    if dtype.is_complex:
        yield SampleInput(make_arg((S, S)), make_arg((S, S)), 0.4j)
        yield SampleInput(make_arg((S, S)), make_arg((S, S)), 1.2 + 0.1j)
        yield SampleInput(make_arg((S, S)), make_arg((S,)), 0.4j)
        yield SampleInput(make_arg((S, S)), make_arg((S, S)), 5.4 + 9j)
        yield SampleInput(make_arg(()), make_arg(()), 0.4j)
        yield SampleInput(make_arg(()), make_arg(()), 6.1 + 0.004j)
        yield SampleInput(make_arg((S, S)), make_arg(()), 0.4j)
        yield SampleInput(make_arg((S, S)), make_arg(()), 1 + 2j)