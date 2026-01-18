import math
from ..eyring import eyring_equation, EyringParam, EyringParamWithUnits
from chempy.util.testing import requires
from chempy.units import allclose, units_library, default_units as u
def test_EyringParam():
    T = 273.15
    k = EyringParam(40000.0, 100.0)(T)
    ref = _kB_over_h * T * math.exp(100.0 / _R) * math.exp(-40000.0 / _R / T)
    assert abs((k - ref) / ref) < 1e-07