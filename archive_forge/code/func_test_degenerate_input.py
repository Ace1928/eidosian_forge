import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start', [np.array([1, 0, 0]), np.array([0, 1])])
@pytest.mark.parametrize('t', [np.array(1), np.array([1]), np.array([[1]]), np.array([[[1]]]), np.array([]), np.linspace(0, 1, 5)])
def test_degenerate_input(self, start, t):
    if np.asarray(t).ndim > 1:
        with pytest.raises(ValueError):
            geometric_slerp(start=start, end=start, t=t)
    else:
        shape = (t.size,) + start.shape
        expected = np.full(shape, start)
        actual = geometric_slerp(start=start, end=start, t=t)
        assert_allclose(actual, expected)
        non_degenerate = geometric_slerp(start=start, end=start[::-1], t=t)
        assert actual.size == non_degenerate.size