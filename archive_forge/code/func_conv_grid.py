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
def conv_grid(n, c, h, w, meta):
    return (ceildiv(n * h * w, meta['BLOCK_M']), ceildiv(c, meta['BLOCK_N']), meta['GROUPS'])