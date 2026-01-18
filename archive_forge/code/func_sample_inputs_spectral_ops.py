import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def sample_inputs_spectral_ops(self, device, dtype, requires_grad=False, **kwargs):
    is_fp16_or_chalf = dtype == torch.complex32 or dtype == torch.half
    if not is_fp16_or_chalf:
        nd_tensor = partial(make_tensor, (S, S + 1, S + 2), device=device, dtype=dtype, requires_grad=requires_grad)
        oned_tensor = partial(make_tensor, (31,), device=device, dtype=dtype, requires_grad=requires_grad)
    else:
        low = None
        high = None
        if self.name in ['fft.hfft', 'fft.irfft', '_refs.fft.hfft', '_refs.fft.irfft']:
            shapes = ((2, 9, 9), (33,))
        elif self.name in ['fft.hfft2', 'fft.irfft2', '_refs.fft.hfft2', '_refs.fft.irfft2']:
            shapes = ((2, 8, 9), (33,))
        elif self.name in ['fft.hfftn', 'fft.irfftn', '_refs.fft.hfftn', '_refs.fft.irfftn']:
            shapes = ((2, 2, 33), (33,))
            low = -1.0
            high = 1.0
        else:
            shapes = ((2, 8, 16), (32,))
        nd_tensor = partial(make_tensor, shapes[0], device=device, low=low, high=high, dtype=dtype, requires_grad=requires_grad)
        oned_tensor = partial(make_tensor, shapes[1], device=device, low=low, high=high, dtype=dtype, requires_grad=requires_grad)
    if self.ndimensional == SpectralFuncType.ND:
        yield SampleInput(nd_tensor(), s=(3, 10) if not is_fp16_or_chalf else (4, 8), dim=(1, 2), norm='ortho')
        yield SampleInput(nd_tensor(), norm='ortho')
        yield SampleInput(nd_tensor(), s=(8,))
        yield SampleInput(oned_tensor())
        yield from (SampleInput(nd_tensor(), dim=dim) for dim in [-1, -2, -3, (0, -1)])
    elif self.ndimensional == SpectralFuncType.TwoD:
        yield SampleInput(nd_tensor(), s=(3, 10) if not is_fp16_or_chalf else (4, 8), dim=(1, 2), norm='ortho')
        yield SampleInput(nd_tensor(), norm='ortho')
        yield SampleInput(nd_tensor(), s=(6, 8) if not is_fp16_or_chalf else (4, 8))
        yield SampleInput(nd_tensor(), dim=0)
        yield SampleInput(nd_tensor(), dim=(0, -1))
        yield SampleInput(nd_tensor(), dim=(-3, -2, -1))
    else:
        yield SampleInput(nd_tensor(), n=10 if not is_fp16_or_chalf else 8, dim=1, norm='ortho')
        yield SampleInput(nd_tensor(), norm='ortho')
        yield SampleInput(nd_tensor(), n=7 if not is_fp16_or_chalf else 8)
        yield SampleInput(oned_tensor())
        yield from (SampleInput(nd_tensor(), dim=dim) for dim in [-1, -2, -3])