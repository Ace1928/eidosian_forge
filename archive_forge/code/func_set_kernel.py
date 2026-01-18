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
def set_kernel(self, kernel: CppKernel):
    """
        Set the kernel under this loop level. No split is allowed under
        this loop level.
        """
    if not self.inner:
        self.kernel = kernel
        loop: Optional[LoopLevel] = self
        assert loop is not None
        if loop.is_reduction():
            loop.reduction_var_map = kernel.reduction_var_map.copy()
            loop = loop.parent
            while loop is not None and loop.is_reduction():
                assert loop.reduction_var_map is not None
                loop.reduction_var_map.update(kernel.reduction_var_map)
                loop = loop.parent
        return
    assert len(self.inner) == 1
    self.inner[0].set_kernel(kernel)