import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
@pytest.mark.parametrize('axis', [None, 0])
@pytest.mark.parametrize('string_dt', ['S', 'U', 'S0', 'U0'])
@pytest.mark.parametrize('arrs', [([0.0],), ([0.0], [1]), ([0], ['string'], [1.0])])
def test_dtype_with_promotion(self, arrs, string_dt, axis):
    res = np.concatenate(arrs, axis=axis, dtype=string_dt, casting='unsafe')
    assert res.dtype == np.array(1.0).astype(string_dt).dtype