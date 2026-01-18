import operator
from numba import njit, literally
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError
from numba.core.extending import lower_builtin
from numba.core.extending import models, register_model
from numba.core.extending import make_attribute_wrapper
from numba.core.extending import type_callable
from numba.core.extending import overload
from numba.core.extending import typeof_impl
import unittest
def test_boxing(self):
    """A test for the boxing logic on unknown type
        """
    Dummy = self.Dummy

    @njit
    def foo(x):
        return Dummy(x)
    with self.assertRaises(TypeError) as raises:
        foo(123)
    self.assertIn('cannot convert native Dummy to Python object', str(raises.exception))