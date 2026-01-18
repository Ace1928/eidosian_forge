import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP
from sympy.utilities.exceptions import ignore_warnings
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.aesaracode import (aesara_code, dim_handling,
def test_aesara_function_multi():
    """ Test aesara_function() with multiple outputs. """
    f = aesara_function_([x, y], [x + y, x - y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1