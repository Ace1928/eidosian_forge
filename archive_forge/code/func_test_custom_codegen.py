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
def test_custom_codegen():
    from sympy.printing.c import C99CodePrinter
    from sympy.functions.elementary.exponential import exp
    printer = C99CodePrinter(settings={'user_functions': {'exp': 'fastexp'}})
    x, y = symbols('x y')
    expr = exp(x + y)
    gen = C99CodeGen(printer=printer, preprocessor_statements=['#include "fastexp.h"'])
    expected = '#include "expr.h"\n#include "fastexp.h"\ndouble expr(double x, double y) {\n   double expr_result;\n   expr_result = fastexp(x + y);\n   return expr_result;\n}\n'
    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include "fastexp.h"')
    expected = '#include "expr.h"\n#include <math.h>\n#include "fastexp.h"\ndouble expr(double x, double y) {\n   double expr_result;\n   expr_result = fastexp(x + y);\n   return expr_result;\n}\n'
    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected