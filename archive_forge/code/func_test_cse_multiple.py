from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
import sympy
from sympy.abc import a, b, n
def test_cse_multiple():
    e1 = a * a
    e2 = a * a + b * b
    e3 = sympy.cse([e1, e2])
    raises(NotImplementedError, lambda: g.llvm_callable([a, b], e3, callback_type='scipy.integrate'))
    f = g.llvm_callable([a, b], e3)
    jit_res = f(0.1, 1.5)
    assert len(jit_res) == 2
    res = eval_cse(e3, {a: 0.1, b: 1.5})
    assert isclose(res[0], jit_res[0])
    assert isclose(res[1], jit_res[1])