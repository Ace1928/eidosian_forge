import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
@overload_method(MyStructType, 'distance')
def ol_distance(self, other):
    if not isinstance(other, MyStructType):
        raise TypingError(f'*other* must be a {MyStructType}; got {other}')

    def impl(self, other):
        return np.linalg.norm(self.vector - other.vector)
    return impl