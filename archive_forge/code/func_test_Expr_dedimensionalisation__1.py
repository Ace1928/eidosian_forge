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
@requires(units_library)
def test_Expr_dedimensionalisation__1():
    cv = _get_cv(u.kelvin, u.gram, u.mol)
    units, expr = cv['Be'].dedimensionalisation(SI_base_registry)
    assert units == [u.kelvin, u.kg / u.mol]
    assert abs(expr.args[0] - 0.806 * 1440) < 1e-14
    assert abs(expr.args[1] - 0.00901218) < 1e-07