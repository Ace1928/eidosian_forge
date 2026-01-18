from io import StringIO
from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
def test_intrinsic_math_codegen():
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import acos, asin, atan, cos, sin, tan
    x = symbols('x')
    name_expr = [('test_abs', Abs(x)), ('test_acos', acos(x)), ('test_asin', asin(x)), ('test_atan', atan(x)), ('test_cos', cos(x)), ('test_cosh', cosh(x)), ('test_log', log(x)), ('test_ln', log(x)), ('test_sin', sin(x)), ('test_sinh', sinh(x)), ('test_sqrt', sqrt(x)), ('test_tan', tan(x)), ('test_tanh', tanh(x))]
    result = codegen(name_expr, 'F95', 'file', header=False, empty=False)
    assert result[0][0] == 'file.f90'
    expected = 'REAL*8 function test_abs(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_abs = abs(x)\nend function\nREAL*8 function test_acos(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_acos = acos(x)\nend function\nREAL*8 function test_asin(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_asin = asin(x)\nend function\nREAL*8 function test_atan(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_atan = atan(x)\nend function\nREAL*8 function test_cos(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_cos = cos(x)\nend function\nREAL*8 function test_cosh(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_cosh = cosh(x)\nend function\nREAL*8 function test_log(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_log = log(x)\nend function\nREAL*8 function test_ln(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_ln = log(x)\nend function\nREAL*8 function test_sin(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_sin = sin(x)\nend function\nREAL*8 function test_sinh(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_sinh = sinh(x)\nend function\nREAL*8 function test_sqrt(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_sqrt = sqrt(x)\nend function\nREAL*8 function test_tan(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_tan = tan(x)\nend function\nREAL*8 function test_tanh(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest_tanh = tanh(x)\nend function\n'
    assert result[0][1] == expected
    assert result[1][0] == 'file.h'
    expected = 'interface\nREAL*8 function test_abs(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_acos(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_asin(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_atan(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_cos(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_cosh(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_log(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_ln(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_sin(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_sinh(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_sqrt(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_tan(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\ninterface\nREAL*8 function test_tanh(x)\nimplicit none\nREAL*8, intent(in) :: x\nend function\nend interface\n'
    assert result[1][1] == expected