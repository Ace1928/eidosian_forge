from sympy.concrete.summations import Sum
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, i, a, b
from sympy.codegen.numpy_nodes import logaddexp
from sympy.printing.numpy import CuPyPrinter, _cupy_known_constants, _cupy_known_functions
from sympy.testing.pytest import skip
from sympy.external import import_module
def test_cupy_known_funcs_consts():
    assert _cupy_known_constants['NaN'] == 'cupy.nan'
    assert _cupy_known_constants['EulerGamma'] == 'cupy.euler_gamma'
    assert _cupy_known_functions['acos'] == 'cupy.arccos'
    assert _cupy_known_functions['log'] == 'cupy.log'