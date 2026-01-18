import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_find_peaks_with_non_default_wavelets(self):
    x = gaussian(200, 2)
    widths = np.array([1, 2, 3, 4])
    a = find_peaks_cwt(x, widths, wavelet=gaussian)
    np.testing.assert_equal(np.array([100]), a)