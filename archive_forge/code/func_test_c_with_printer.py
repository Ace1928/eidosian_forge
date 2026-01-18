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
def test_c_with_printer():
    from sympy.printing.c import C99CodePrinter

    class CustomPrinter(C99CodePrinter):

        def _print_Pow(self, expr):
            return 'fastpow({}, {})'.format(self._print(expr.base), self._print(expr.exp))
    x = symbols('x')
    expr = x ** 3
    expected = [('file.c', '#include "file.h"\n#include <math.h>\ndouble test(double x) {\n   double test_result;\n   test_result = fastpow(x, 3);\n   return test_result;\n}\n'), ('file.h', '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\ndouble test(double x);\n#endif\n')]
    result = codegen(('test', expr), 'C', 'file', header=False, empty=False, printer=CustomPrinter())
    assert result == expected