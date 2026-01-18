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
def test_PiecewisePoly():
    Poly = mk_Poly('temperature')
    p1 = Poly([0, 1, 0.1])
    assert p1.eval_poly({'temperature': 10}) == 2
    p2 = Poly([0, 3, -0.1])
    assert p2.eval_poly({'temperature': 10}) == 2
    TPiecewisePoly = mk_PiecewisePoly('temperature')
    tpwp = TPiecewisePoly.from_polynomials([(0, 10), (10, 20)], [p1, p2])
    assert tpwp.eval_poly({'temperature': 5}) == 1.5
    assert tpwp.eval_poly({'temperature': 15}) == 1.5
    assert tpwp.parameter_keys == ('temperature',)
    with pytest.raises(ValueError):
        tpwp.eval_poly({'temperature': 21})