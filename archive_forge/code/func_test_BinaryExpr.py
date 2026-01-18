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
def test_BinaryExpr():
    Poly = Expr.from_callback(_poly, parameter_keys=('x',), argument_names=('x0', Ellipsis))
    p1 = Poly([1, 2, 3])
    p2 = Poly([2, 3, 4])
    assert p1({'x': 5}) == 14
    assert p2({'x': 5}) == 15
    assert (p1 + p2)({'x': 5}) == 14 + 15
    assert (p1 - p2)({'x': 5}) == 14 - 15
    assert (p1 * p2)({'x': 5}) == 14 * 15
    assert (p1 / p2)({'x': 5}) == 14 / 15
    assert (p1 + 2)({'x': 5}) == 14 + 2
    assert (p1 - 2)({'x': 5}) == 14 - 2
    assert (p1 * 2)({'x': 5}) == 14 * 2
    assert (p1 / 2)({'x': 5}) == 14 / 2
    assert (2 + p1)({'x': 5}) == 2 + 14
    assert (2 - p1)({'x': 5}) == 2 - 14
    assert (2 * p1)({'x': 5}) == 2 * 14
    assert (2 / p1)({'x': 5}) == 2 / 14
    assert p1 + 0 == p1
    assert p1 * 1 == p1
    assert p1 + p2 == p1 + p2
    assert p1 + p2 * 1 == p1 + p2 + 0
    assert --p1 == p1