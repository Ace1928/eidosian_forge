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
def test_Expr_symbolic():
    import sympy
    cv = _get_cv()
    R, T = sympy.symbols('R T')
    sexpr = cv['Be']({'temperature': T, 'molar_gas_constant': R}, backend=sympy)
    assert sexpr.free_symbols == set([T, R])