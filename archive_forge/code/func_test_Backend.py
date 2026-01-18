from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_Backend():
    b = Backend()
    with pytest.raises(ValueError):
        b.exp(-3 * u.metre)
    assert abs(b.exp(1234 * u.metre / u.kilometre) - b.exp(1.234)) < 1e-14