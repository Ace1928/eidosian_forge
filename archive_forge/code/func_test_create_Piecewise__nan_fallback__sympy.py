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
def test_create_Piecewise__nan_fallback__sympy():
    import sympy as sp
    TPw = create_Piecewise('Tmpr', nan_fallback=True)
    pw = TPw([0, 42, 10, 43, 20])
    x = sp.Symbol('x')
    res = pw({'Tmpr': x}, backend=sp)
    assert isinstance(res, sp.Piecewise)
    assert res.args[0][0] == 42
    assert res.args[0][1] == sp.And(0 <= x, x <= 10)
    assert res.args[1][0] == 43
    assert res.args[1][1] == sp.And(10 <= x, x <= 20)
    assert res.args[2][0].name.lower() == 'nan'
    assert res.args[2][1] == True