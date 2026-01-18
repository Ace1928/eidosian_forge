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
def test_check_case_false_positive():
    x1 = symbols('x')
    x2 = symbols('x', my_assumption=True)
    try:
        codegen(('test', x1 * x2), 'f95', 'prefix')
    except CodeGenError as e:
        if e.args[0].startswith('Fortran ignores case.'):
            raise AssertionError('This exception should not be raised!')