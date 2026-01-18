import pytest
from .._solution import Solution, QuantityDict
from ..util.testing import requires
from ..units import magnitude, units_library, to_unitless, default_units as u
@requires(units_library)
def test_Solution__add():
    s1, s2 = _get_s1_s2()
    s3 = s1 + s2
    assert abs(to_unitless(s3.volume - 0.0004 * u.m ** 3, u.dm3)) < 1e-15