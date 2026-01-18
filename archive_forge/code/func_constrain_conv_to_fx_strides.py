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
def constrain_conv_to_fx_strides(fx_node, *args, **kwargs):
    assert fx_node.target == torch.ops.aten.convolution.default
    if V.graph.layout_opt:
        return (args, kwargs)
    else:
        return constrain_to_fx_strides(fx_node, *args, **kwargs)