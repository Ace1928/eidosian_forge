import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start, end', [(np.zeros((1, 3)), np.ones((1, 3))), (np.zeros((1, 3)), np.ones(3)), (np.zeros(1), np.ones((3, 1)))])
def test_input_shape_flat(self, start, end):
    with pytest.raises(ValueError, match='one-dimensional'):
        geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))