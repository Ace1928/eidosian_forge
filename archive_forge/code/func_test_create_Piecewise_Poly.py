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
def test_create_Piecewise_Poly():
    PolyT = create_Poly('Tmpr')
    p1 = PolyT([1, 0.1])
    assert p1({'Tmpr': 10}) == 2
    p2 = PolyT([3, -0.1])
    assert p2({'Tmpr': 10}) == 2
    PiecewiseT = create_Piecewise('Tmpr')
    pw = PiecewiseT([0, p1, 10, p2, 20])
    assert pw({'Tmpr': 5}) == 1.5
    assert pw({'Tmpr': 15}) == 1.5
    assert pw.parameter_keys == ('Tmpr',)
    with pytest.raises(ValueError):
        pw({'Tmpr': 21})