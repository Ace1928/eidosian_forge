import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def get_shape_classes(self, name):
    """Instead of the shape tuple, return tuple of int, where
        each int is the corresponding class index of the size object.
        Unknown shapes are given class index -1. Return empty tuple
        if the input name is a scalar variable.
        """
    if isinstance(name, ir.Var):
        name = name.name
    typ = self.typemap[name] if name in self.typemap else None
    if not isinstance(typ, (types.BaseTuple, types.SliceType, types.ArrayCompatible)):
        return []
    if isinstance(typ, types.ArrayCompatible) and typ.ndim == 0:
        return []
    names = self._get_names(name)
    inds = tuple((self._get_ind(name) for name in names))
    return inds