import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def get_equiv_const(self, obj):
    """If the given object is equivalent to a constant scalar,
        return the scalar value, or None otherwise.
        """
    names = self._get_names(obj)
    if len(names) != 1:
        return None
    return super(ShapeEquivSet, self).get_equiv_const(names[0])