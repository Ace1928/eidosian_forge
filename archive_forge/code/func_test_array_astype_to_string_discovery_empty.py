import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize('dt', ['S', 'U'])
def test_array_astype_to_string_discovery_empty(dt):
    arr = np.array([''], dtype=object)
    assert arr.astype(dt).dtype.itemsize == np.dtype(f'{dt}1').itemsize
    assert np.can_cast(arr, dt, casting='unsafe')
    assert not np.can_cast(arr, dt, casting='same_kind')
    assert np.can_cast('O', dt, casting='unsafe')