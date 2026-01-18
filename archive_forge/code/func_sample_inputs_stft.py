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
def sample_inputs_stft(op_info, device, dtype, requires_grad, **kwargs):

    def mt(shape, **kwargs):
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)
    yield SampleInput(mt(100), n_fft=10, return_complex=True)
    yield SampleInput(mt(100), n_fft=10, return_complex=False)
    if dtype.is_complex:
        yield SampleInput(mt(100), n_fft=10)
    for center in [False, True]:
        yield SampleInput(mt(10), n_fft=7, center=center, return_complex=True)
        yield SampleInput(mt((10, 100)), n_fft=16, hop_length=4, center=center, return_complex=True)
    window = mt(16, low=0.5, high=2.0)
    yield SampleInput(mt((2, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True, center=center))
    yield SampleInput(mt((3, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True, center=center))
    if not dtype.is_complex:
        yield SampleInput(mt((10, 100)), n_fft=16, window=window, onesided=False, return_complex=True)