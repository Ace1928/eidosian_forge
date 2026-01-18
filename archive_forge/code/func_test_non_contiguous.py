import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_non_contiguous(self):
    """
        Test with non-C-contiguous input arrays.
        """
    x = np.repeat([0, 100, 50], 4)
    peaks = np.repeat([1], 3)
    result = peak_widths(x[::4], peaks[::3])
    assert_equal(result, [0.75, 75, 0.75, 1.5])