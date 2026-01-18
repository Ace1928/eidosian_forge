import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start, end', [(np.array([]), np.array([]))])
def test_input_at_least1d(self, start, end):
    with pytest.raises(ValueError, match='at least two-dim'):
        geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))