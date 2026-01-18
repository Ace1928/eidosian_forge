import math
from chempy.chemistry import Reaction
from chempy.util.testing import requires
from chempy.units import units_library, allclose, default_units as u
from ..arrhenius import arrhenius_equation, ArrheniusParam, ArrheniusParamWithUnits
def test_ArrheniusParam():
    k = ArrheniusParam(_A1, _Ea1)(_T1)
    assert abs((k - _k1) / _k1) < 0.0001