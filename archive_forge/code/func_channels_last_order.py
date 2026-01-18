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
def channels_last_order(rank):
    order = list(reversed(range(rank)))
    order.insert(1, order.pop(-1))
    return order