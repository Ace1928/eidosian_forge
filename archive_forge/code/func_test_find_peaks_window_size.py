import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_find_peaks_window_size(self):
    """
        Verify that window_size is passed correctly to private function and
        affects the result.
        """
    sigmas = [2.0, 2.0]
    num_points = 1000
    test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
    widths = np.arange(0.1, max(sigmas), 0.2)
    noise_amp = 0.05
    np.random.seed(18181911)
    test_data += (np.random.rand(num_points) - 0.5) * (2 * noise_amp)
    test_data[250:320] -= 1
    found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3, min_length=None, window_size=None)
    with pytest.raises(AssertionError):
        assert found_locs.size == act_locs.size
    found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3, min_length=None, window_size=20)
    assert found_locs.size == act_locs.size