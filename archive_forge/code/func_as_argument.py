from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def as_argument(self, builder, value):
    inner = self.get(builder, value)
    return self._actual_model.as_argument(builder, inner)