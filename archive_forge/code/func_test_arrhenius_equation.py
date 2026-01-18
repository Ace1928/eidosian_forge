import math
from chempy.chemistry import Reaction
from chempy.util.testing import requires
from chempy.units import units_library, allclose, default_units as u
from ..arrhenius import arrhenius_equation, ArrheniusParam, ArrheniusParamWithUnits
def test_arrhenius_equation():
    assert abs(arrhenius_equation(3, 831.4472, 100) - 3 / 2.7182818) < 1e-07