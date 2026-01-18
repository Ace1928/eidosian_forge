import pytest
from .._solution import Solution, QuantityDict
from ..util.testing import requires
from ..units import magnitude, units_library, to_unitless, default_units as u
@requires(units_library)
def test_Solution__withdraw():
    s1, s2 = _get_s1_s2()
    s3 = s1 + s2
    s4 = s3.withdraw(0.2 * u.dm3)
    assert s4 == s3
    assert s4 == (s1 + s2).withdraw(0.2 * u.dm3)
    assert s4 != s1 + s2