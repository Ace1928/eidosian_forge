import pytest
from numpy.testing import assert_almost_equal
from ..mriutils import MRIError, calculate_dwell_time
def test_calculate_dwell_time():
    assert_almost_equal(calculate_dwell_time(3.3, 2, 3), 3.3 / (42.576 * 3.4 * 3 * 3))
    assert_almost_equal(calculate_dwell_time(3.3, 1, 3), 0)
    with pytest.raises(MRIError):
        calculate_dwell_time(3.3, 0, 3.0)
    with pytest.raises(MRIError):
        calculate_dwell_time(3.3, 2, -0.1)