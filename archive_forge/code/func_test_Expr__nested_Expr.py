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
@requires(parsing_library)
def test_Expr__nested_Expr():
    Poly = Expr.from_callback(_poly, parameter_keys=('x',), argument_names=('x0', Ellipsis))
    T = Poly([3, 7, 5])
    cv = _get_cv()
    _ref = 0.8108020083055849
    args = {'temperature': T, 'x': (273.15 - 7) / 5 + 3, 'molar_gas_constant': 8.3145}
    assert abs(cv['Al'](args) - _ref) < 1e-14
    Al2 = cv['Al'] / 2
    assert abs(Al2(args) - _ref / 2) < 1e-14