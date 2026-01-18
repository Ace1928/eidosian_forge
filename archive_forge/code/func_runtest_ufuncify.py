import sympy
import tempfile
import os
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip
def runtest_ufuncify(language, backend):
    has_module('numpy')
    a, b, c = symbols('a b c')
    fabc = ufuncify([a, b, c], a * b + c, backend=backend)
    facb = ufuncify([a, c, b], a * b + c, backend=backend)
    grid = numpy.linspace(-2, 2, 50)
    b = numpy.linspace(-5, 4, 50)
    c = numpy.linspace(-1, 1, 50)
    expected = grid * b + c
    numpy.testing.assert_allclose(fabc(grid, b, c), expected)
    numpy.testing.assert_allclose(facb(grid, c, b), expected)