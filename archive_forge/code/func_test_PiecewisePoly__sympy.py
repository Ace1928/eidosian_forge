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
def test_PiecewisePoly__sympy():
    import sympy as sp
    Poly = mk_Poly('T')
    p1 = Poly([0, 1, 0.1])
    p2 = Poly([0, 3, -0.1])
    TPiecewisePoly = mk_PiecewisePoly('temperature')
    tpwp = TPiecewisePoly([2, 2, 0, 10, 2, 10, 20, 0, 1, 0.1, 0, 3, -0.1])
    x = sp.Symbol('x')
    res = tpwp.eval_poly({'temperature': x}, backend=sp)
    assert isinstance(res, sp.Piecewise)
    assert res.args[0][0] == 1 + 0.1 * x
    assert res.args[0][1] == sp.And(0 <= x, x <= 10)
    assert res.args[1][0] == 3 - 0.1 * x
    assert res.args[1][1] == sp.And(10 <= x, x <= 20)
    with pytest.raises(ValueError):
        tpwp.from_polynomials([(0, 10), (10, 20)], [p1, p2])