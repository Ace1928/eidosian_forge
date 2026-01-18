from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def init_call_helper(self, builder):
    """
        Initialize and return a call helper object for the given builder.
        """
    ch = self._make_call_helper(builder)
    builder.__call_helper = ch
    return ch