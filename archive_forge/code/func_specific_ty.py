import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
def specific_ty(z):
    return types.literal(z) if types.maybe_literal(z) else typeof(z)