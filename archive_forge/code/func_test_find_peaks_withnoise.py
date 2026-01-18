import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_find_peaks_withnoise(self):
    """
        Verify that peak locations are (approximately) found
        for a series of gaussians with added noise.
        """
    sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
    num_points = 500
    test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
    widths = np.arange(0.1, max(sigmas))
    noise_amp = 0.07
    np.random.seed(18181911)
    test_data += (np.random.rand(num_points) - 0.5) * (2 * noise_amp)
    found_locs = find_peaks_cwt(test_data, widths, min_length=15, gap_thresh=1, min_snr=noise_amp / 5)
    np.testing.assert_equal(len(found_locs), len(act_locs), 'Different number' + 'of peaks found than expected')
    diffs = np.abs(found_locs - act_locs)
    max_diffs = np.array(sigmas) / 5
    np.testing.assert_array_less(diffs, max_diffs, 'Maximum location differed' + 'by more than %s' % max_diffs)