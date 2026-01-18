from sympy.concrete.summations import Sum
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, i, a, b
from sympy.codegen.numpy_nodes import logaddexp
from sympy.printing.numpy import CuPyPrinter, _cupy_known_constants, _cupy_known_functions
from sympy.testing.pytest import skip
from sympy.external import import_module
def test_cupy_print_methods():
    prntr = CuPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')