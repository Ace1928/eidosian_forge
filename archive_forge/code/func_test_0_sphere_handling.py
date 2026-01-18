import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start, end', [(np.array([1]), np.array([0])), (np.array([0]), np.array([1])), (np.array([-17.7]), np.array([165.9]))])
def test_0_sphere_handling(self, start, end):
    with pytest.raises(ValueError, match='at least two-dim'):
        _ = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 4))