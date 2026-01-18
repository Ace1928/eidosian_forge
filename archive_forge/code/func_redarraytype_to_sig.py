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
def redarraytype_to_sig(redarraytyp):
    """Given a reduction array type, find the type of the reduction argument to the gufunc.
    """
    assert isinstance(redarraytyp, types.npytypes.Array)
    return types.npytypes.Array(redarraytyp.dtype, redarraytyp.ndim, redarraytyp.layout)