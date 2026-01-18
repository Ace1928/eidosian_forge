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
def sample_inputs_upsample_aa(mode, self, device, dtype, requires_grad, **kwargs):
    N = 6
    C = 3
    H = 10
    W = 20
    S = 3
    L = 5
    input_tensor = make_tensor(torch.Size([N, C, H, W]), device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=False, scale_factors=None)
    yield SampleInput(input_tensor, output_size=torch.Size([L, L]), align_corners=False, scale_factors=None)
    yield SampleInput(input_tensor, output_size=None, align_corners=False, scale_factors=[1.7, 0.9])
    yield SampleInput(input_tensor, output_size=None, align_corners=True, scale_factors=[0.8, 1.0])
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=False, scales_h=None, scales_w=None)
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=False, scales_h=1.7, scales_w=0.9)
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=True, scales_h=1.7, scales_w=0.9)