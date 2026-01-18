from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_bad_callback():
    e = a
    raises(ValueError, lambda: g.llvm_callable([a], e, callback_type='bad_callback'))