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
def error_inputs_cov(op_info, device, **kwargs):
    a = torch.rand(S, device=device)
    yield ErrorInput(SampleInput(torch.rand(S, S, S, device=device)), error_regex='expected input to have two or fewer dimensions')
    yield ErrorInput(SampleInput(a, fweights=torch.rand(S, S, device=device)), error_regex='expected fweights to have one or fewer dimensions')
    yield ErrorInput(SampleInput(a, aweights=torch.rand(S, S, device=device)), error_regex='expected aweights to have one or fewer dimensions')
    yield ErrorInput(SampleInput(a, fweights=torch.rand(S, device=device)), error_regex='expected fweights to have integral dtype')
    yield ErrorInput(SampleInput(a, aweights=torch.tensor([1, 1], device=device)), error_regex='expected aweights to have floating point dtype')
    yield ErrorInput(SampleInput(a, fweights=torch.tensor([1], device=device)), error_regex='expected fweights to have the same numel')
    yield ErrorInput(SampleInput(a, aweights=torch.rand(1, device=device)), error_regex='expected aweights to have the same numel')
    yield ErrorInput(SampleInput(a, fweights=torch.tensor([-1, -2, -3, -4, -5], device=device)), error_regex='fweights cannot be negative')
    yield ErrorInput(SampleInput(a, aweights=torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device=device)), error_regex='aweights cannot be negative')