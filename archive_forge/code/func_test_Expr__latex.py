import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
@requires('sympy')
def test_Expr__latex():
    Poly = Expr.from_callback(_poly, parameter_keys=('x',), argument_names=('x0', Ellipsis))
    p = Poly([1, 2, 3, 4])
    import sympy
    t = sympy.Symbol('t')
    ref = sympy.latex((2 + 3 * (t - 1) + 4 * (t - 1) ** 2).simplify())
    assert p.latex({'x': 't'}) == ref
    TE = Poly([3, 7, 5])
    cv_Al = _get_cv()['Al']
    T, E, R, m = sympy.symbols('T E R m')
    _TE = 7 + 5 * (E - 3)
    ref = sympy.latex((3 * R * (_TE / (2 * T)) ** 2 * sympy.sinh(_TE / (2 * T)) ** (-2) / m).simplify())
    cv_Al.unique_keys = ('TE_Al', 'm_Al')
    res = cv_Al.latex({'TE_Al': TE, 'temperature': 'T', 'x': 'E', 'molar_gas_constant': 'R', 'm_Al': 'm'})
    assert res == ref
    X = sympy.symbols('X')
    _TE2 = 7 + 5 * (X - 3)
    ref2 = sympy.latex((3 * R * (_TE2 / (2 * T)) ** 2 * sympy.sinh(_TE2 / (2 * T)) ** (-2) / m).simplify())
    res2 = cv_Al.latex({'TE_Al': TE, 'temperature': 'T', 'molar_gas_constant': 'R', 'm_Al': 'm'}, default=lambda k: k.upper())
    assert res2 == ref2