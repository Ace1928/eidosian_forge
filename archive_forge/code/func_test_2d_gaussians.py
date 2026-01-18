import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_2d_gaussians(self):
    sigmas = [1.0, 2.0, 10.0]
    test_data, act_locs = _gen_gaussians_even(sigmas, 100)
    rot_factor = 20
    rot_range = np.arange(0, len(test_data)) - rot_factor
    test_data_2 = np.vstack([test_data, test_data[rot_range]])
    rel_max_rows, rel_max_cols = argrelmax(test_data_2, axis=1, order=1)
    for rw in range(0, test_data_2.shape[0]):
        inds = rel_max_rows == rw
        assert_(len(rel_max_cols[inds]) == len(act_locs))
        assert_((act_locs == rel_max_cols[inds] - rot_factor * rw).all())