import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_find_peaks_with_one_width(self):
    """
        Verify that the `width` argument
        in `find_peaks_cwt` can be a float
        """
    xs = np.arange(0, np.pi, 0.05)
    test_data = np.sin(xs)
    widths = 1
    found_locs = find_peaks_cwt(test_data, widths)
    np.testing.assert_equal(found_locs, 32)