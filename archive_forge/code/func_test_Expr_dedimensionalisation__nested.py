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
def test_Expr_dedimensionalisation__nested():
    Poly = Expr.from_callback(_poly, parameter_keys=('E',), argument_names=('x0', Ellipsis))
    TE = Poly([3 * u.J, 7 * u.K, 5 * u.K / u.J])
    TE = Poly([0.7170172084130019 * u.cal, 12.6 * u.Rankine, 5 * u.K / u.J]) * 0.806 * 428 / 273.15
    _ref = 0.8108020083055849
    cv_Al = _get_cv(u.kelvin, u.gram, u.mol)['Al']
    cv_Al_units, Al_dedim = cv_Al.dedimensionalisation(SI_base_registry, {'einstein_temperature': TE})
    assert abs(Al_dedim({'temperature': 273.15, 'E': (273.15 - 7) / 5 + 3, 'molar_gas_constant': 8.3145}) - _ref * 1000) < 1e-14