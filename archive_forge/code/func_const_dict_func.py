import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
def const_dict_func():
    """
            Dictionary update between two constant
            dictionaries. This verifies d2 doesn't
            get incorrectly removed.
            """
    d1 = {'a': 1, 'b': 2, 'c': 3}
    d2 = {'d': 4, 'e': 4}
    check_before(d1)
    d1.update(d2)
    check_after(d1)
    if len(d1) > 4:
        return d2
    return d1