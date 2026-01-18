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
def test_Expr__from_callback():

    def two_dim_gauss(args, x, y, backend=None):
        A, x0, y0, sx, sy = args
        xp, yp = (x - x0, y - y0)
        vx, vy = (2 * sx ** 2, 2 * sy ** 2)
        return A * backend.exp(-(xp ** 2 / vx + yp ** 2 / vy))
    TwoDimGauss = Expr.from_callback(two_dim_gauss, parameter_keys=('x', 'y'), nargs=5)
    with pytest.raises(ValueError):
        TwoDimGauss([1, 2])
    args = [3, 2, 1, 4, 5]
    g1 = TwoDimGauss(args)
    ref = two_dim_gauss(args, 6, 7, math)
    assert abs(g1({'x': 6, 'y': 7}) - ref) < 1e-14