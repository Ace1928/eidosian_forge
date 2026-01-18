from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def i0(x: ArrayLike):
    return torch.special.i0(x)