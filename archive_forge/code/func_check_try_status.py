from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def check_try_status(self, builder):
    try_state_ptr = self._get_try_state(builder)
    try_depth = builder.load(try_state_ptr)
    in_try = builder.icmp_unsigned('>', try_depth, try_depth.type(0))
    excinfoptr = self._get_excinfo_argument(builder.function)
    excinfo = builder.load(excinfoptr)
    return TryStatus(in_try=in_try, excinfo=excinfo)