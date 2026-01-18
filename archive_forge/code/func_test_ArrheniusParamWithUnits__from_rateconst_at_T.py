import math
from chempy.chemistry import Reaction
from chempy.util.testing import requires
from chempy.units import units_library, allclose, default_units as u
from ..arrhenius import arrhenius_equation, ArrheniusParam, ArrheniusParamWithUnits
@requires(units_library)
def test_ArrheniusParamWithUnits__from_rateconst_at_T():
    _2 = _get_ref2_units()
    apu = ArrheniusParamWithUnits.from_rateconst_at_T(_2.Ea, (_2.T, _2.k))
    assert allclose(apu(_2.T), _2.k)