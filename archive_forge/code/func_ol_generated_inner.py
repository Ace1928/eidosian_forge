from numba import int32, int64
from numba import jit
from numba.core import types
from numba.extending import overload
from numba.tests.support import TestCase, tag
import unittest
@overload(generated_inner)
def ol_generated_inner(x, y=5, z=6):
    if isinstance(x, types.Complex):

        def impl(x, y=5, z=6):
            return (x + y, z)
    else:

        def impl(x, y=5, z=6):
            return (x - y, z)
    return impl