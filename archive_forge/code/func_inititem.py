import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
def inititem(self, idx, val, incref=True):
    ptr = self._gep(idx)
    data_item = self._datamodel.as_data(self._builder, val)
    self._builder.store(data_item, ptr)
    if incref:
        self.incref_value(val)