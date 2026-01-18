from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_logspace_from_lin():
    ls = logspace_from_lin(2 * u.second, 3 * u.second)
    assert abs(to_unitless(ls[0], u.hour) - 2 / 3600.0) < 1e-15
    assert abs(to_unitless(ls[-1], u.hour) - 3 / 3600.0) < 1e-15