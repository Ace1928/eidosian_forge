from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library, 'numpy')
def test_Backend__numpy():
    import numpy as np
    b = Backend(np)
    b.sum([1000 * u.metre / u.kilometre, 1], axis=0) == 2.0
    with pytest.raises(AttributeError):
        b.Piecewise