from __future__ import annotations
import functools
import logging
from typing import cast, List, Optional, Sequence, Tuple, TypedDict
import torch
from .. import config, ir
from ..ir import TensorBox
from ..lowering import (
from ..select_algorithm import (
from ..utils import (
from ..virtualized import V
from .mm_common import filtered_configs
def conv1x1_via_mm(x, w, *, out):
    w = torch.squeeze(torch.squeeze(w, -1), -1)
    return torch.matmul(x.permute(0, 2, 3, 1), w.permute(1, 0), out=out.permute(0, 2, 3, 1))