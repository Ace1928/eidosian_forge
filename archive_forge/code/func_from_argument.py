from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def from_argument(self, builder, value):
    res = self._actual_model.from_argument(builder, value)
    return self.set(builder, self.make_uninitialized(), res)