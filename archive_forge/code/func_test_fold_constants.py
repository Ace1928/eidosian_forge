from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_fold_constants():
    assert abs(fold_constants(dc.pi) - np.pi) < 1e-15