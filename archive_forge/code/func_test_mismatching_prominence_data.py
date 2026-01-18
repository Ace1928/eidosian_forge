import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_mismatching_prominence_data(self):
    """Test with mismatching peak and / or prominence data."""
    x = [0, 1, 0]
    peak = [1]
    for i, (prominences, left_bases, right_bases) in enumerate([((1.0,), (-1,), (2,)), ((1.0,), (0,), (3,)), ((1.0,), (2,), (0,)), ((1.0, 1.0), (0, 0), (2, 2)), ((1.0, 1.0), (0,), (2,)), ((1.0,), (0, 0), (2,)), ((1.0,), (0,), (2, 2))]):
        prominence_data = (np.array(prominences, dtype=np.float64), np.array(left_bases, dtype=np.intp), np.array(right_bases, dtype=np.intp))
        if i < 3:
            match = 'prominence data is invalid for peak'
        else:
            match = 'arrays in `prominence_data` must have the same shape'
        with raises(ValueError, match=match):
            peak_widths(x, peak, prominence_data=prominence_data)