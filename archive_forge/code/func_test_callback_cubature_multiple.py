from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_callback_cubature_multiple():
    e1 = a * a
    e2 = a * a + b * b
    e3 = sympy.cse([e1, e2, 4 * e2])
    f = g.llvm_callable([a, b], e3, callback_type='cubature')
    ndim = 2
    outdim = 3
    m = ctypes.c_int(ndim)
    fdim = ctypes.c_int(outdim)
    array_type = ctypes.c_double * ndim
    out_array_type = ctypes.c_double * outdim
    inp = {a: 0.2, b: 1.5}
    array = array_type(inp[a], inp[b])
    out_array = out_array_type()
    jit_ret = f(m, array, None, fdim, out_array)
    assert jit_ret == 0
    res = eval_cse(e3, inp)
    assert isclose(out_array[0], res[0])
    assert isclose(out_array[1], res[1])
    assert isclose(out_array[2], res[2])