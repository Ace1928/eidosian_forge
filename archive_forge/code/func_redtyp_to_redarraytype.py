import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def redtyp_to_redarraytype(redtyp):
    """Go from a reducation variable type to a reduction array type used to hold
       per-worker results.
    """
    redarrdim = 1
    if isinstance(redtyp, types.npytypes.Array):
        redarrdim += redtyp.ndim
        redtyp = redtyp.dtype
    return types.npytypes.Array(redtyp, redarrdim, 'C')