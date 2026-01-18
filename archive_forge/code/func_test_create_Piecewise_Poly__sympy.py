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
def test_create_Piecewise_Poly__sympy():
    import sympy as sp
    Poly = create_Poly('Tmpr')
    p1 = Poly([1, 0.1])
    p2 = Poly([3, -0.1])
    TPw = create_Piecewise('Tmpr')
    pw = TPw([0, p1, 10, p2, 20])
    x = sp.Symbol('x')
    res = pw({'Tmpr': x}, backend=sp)
    assert isinstance(res, sp.Piecewise)
    assert res.args[0][0] == 1 + 0.1 * x
    assert res.args[0][1] == sp.And(0 <= x, x <= 10)
    assert res.args[1][0] == 3 - 0.1 * x
    assert res.args[1][1] == sp.And(10 <= x, x <= 20)