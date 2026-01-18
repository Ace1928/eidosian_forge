import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
def is_supported_cmp(self, node: torch.fx.Node):

    def get_node_dtype(node):
        if type(node) == torch.fx.Node:
            opt_ctx: OptimizationContext = get_current_node_opt_ctx()
            return opt_ctx.dtype if opt_ctx else None
        else:
            return None

    def get_cmp_dtypes(node: torch.fx.Node):
        return (get_node_dtype(node.args[-2]), get_node_dtype(node.args[-1]))
    assert len(node.args) >= 2
    if type(node.args[-1]) in [int, float]:
        return True
    if type(node.args[-2]) in [int, float]:
        return False
    left_dtype, right_dtype = get_cmp_dtypes(node)
    if left_dtype is None or right_dtype is None:
        return True
    else:
        return left_dtype == right_dtype