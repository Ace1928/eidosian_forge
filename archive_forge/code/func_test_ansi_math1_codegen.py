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
def test_ansi_math1_codegen():
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import acos, asin, atan, cos, sin, tan
    x = symbols('x')
    name_expr = [('test_fabs', Abs(x)), ('test_acos', acos(x)), ('test_asin', asin(x)), ('test_atan', atan(x)), ('test_ceil', ceiling(x)), ('test_cos', cos(x)), ('test_cosh', cosh(x)), ('test_floor', floor(x)), ('test_log', log(x)), ('test_ln', log(x)), ('test_sin', sin(x)), ('test_sinh', sinh(x)), ('test_sqrt', sqrt(x)), ('test_tan', tan(x)), ('test_tanh', tanh(x))]
    result = codegen(name_expr, 'C89', 'file', header=False, empty=False)
    assert result[0][0] == 'file.c'
    assert result[0][1] == '#include "file.h"\n#include <math.h>\ndouble test_fabs(double x) {\n   double test_fabs_result;\n   test_fabs_result = fabs(x);\n   return test_fabs_result;\n}\ndouble test_acos(double x) {\n   double test_acos_result;\n   test_acos_result = acos(x);\n   return test_acos_result;\n}\ndouble test_asin(double x) {\n   double test_asin_result;\n   test_asin_result = asin(x);\n   return test_asin_result;\n}\ndouble test_atan(double x) {\n   double test_atan_result;\n   test_atan_result = atan(x);\n   return test_atan_result;\n}\ndouble test_ceil(double x) {\n   double test_ceil_result;\n   test_ceil_result = ceil(x);\n   return test_ceil_result;\n}\ndouble test_cos(double x) {\n   double test_cos_result;\n   test_cos_result = cos(x);\n   return test_cos_result;\n}\ndouble test_cosh(double x) {\n   double test_cosh_result;\n   test_cosh_result = cosh(x);\n   return test_cosh_result;\n}\ndouble test_floor(double x) {\n   double test_floor_result;\n   test_floor_result = floor(x);\n   return test_floor_result;\n}\ndouble test_log(double x) {\n   double test_log_result;\n   test_log_result = log(x);\n   return test_log_result;\n}\ndouble test_ln(double x) {\n   double test_ln_result;\n   test_ln_result = log(x);\n   return test_ln_result;\n}\ndouble test_sin(double x) {\n   double test_sin_result;\n   test_sin_result = sin(x);\n   return test_sin_result;\n}\ndouble test_sinh(double x) {\n   double test_sinh_result;\n   test_sinh_result = sinh(x);\n   return test_sinh_result;\n}\ndouble test_sqrt(double x) {\n   double test_sqrt_result;\n   test_sqrt_result = sqrt(x);\n   return test_sqrt_result;\n}\ndouble test_tan(double x) {\n   double test_tan_result;\n   test_tan_result = tan(x);\n   return test_tan_result;\n}\ndouble test_tanh(double x) {\n   double test_tanh_result;\n   test_tanh_result = tanh(x);\n   return test_tanh_result;\n}\n'
    assert result[1][0] == 'file.h'
    assert result[1][1] == '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\ndouble test_fabs(double x);\ndouble test_acos(double x);\ndouble test_asin(double x);\ndouble test_atan(double x);\ndouble test_ceil(double x);\ndouble test_cos(double x);\ndouble test_cosh(double x);\ndouble test_floor(double x);\ndouble test_log(double x);\ndouble test_ln(double x);\ndouble test_sin(double x);\ndouble test_sinh(double x);\ndouble test_sqrt(double x);\ndouble test_tan(double x);\ndouble test_tanh(double x);\n#endif\n'