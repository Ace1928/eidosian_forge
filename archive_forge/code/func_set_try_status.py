from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def set_try_status(self, builder):
    try_state_ptr = self._get_try_state(builder)
    old = builder.load(try_state_ptr)
    new = builder.add(old, old.type(1))
    builder.store(new, try_state_ptr)