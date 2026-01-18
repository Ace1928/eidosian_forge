from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_cse():
    e = a * a + b * b + sympy.exp(-a * a - b * b)
    e2 = sympy.cse(e)
    f = g.llvm_callable([a, b], e2)
    res = float(e.subs({a: 2.3, b: 0.1}).evalf())
    jit_res = f(2.3, 0.1)
    assert isclose(jit_res, res)