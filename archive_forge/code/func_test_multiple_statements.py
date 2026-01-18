from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_multiple_statements():
    e = [[(b, 4.0 * a)], [b + 5]]
    f = g.llvm_callable([a], e)
    b_val = e[0][0][1].subs({a: 1.5})
    res = float(e[1][0].subs({b: b_val}).evalf())
    jit_res = f(1.5)
    assert isclose(jit_res, res)
    f_callback = g.llvm_callable([a], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    array = array_type(1.5)
    jit_callback_res = f_callback(m, array)
    assert isclose(jit_callback_res, res)