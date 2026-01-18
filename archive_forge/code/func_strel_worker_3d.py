import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def strel_worker_3d(self, fn, func):
    matlab_masks = np.load(fetch(fn))
    k = 0
    for arrname in sorted(matlab_masks):
        expected_mask = matlab_masks[arrname]
        actual_mask = func(k)
        if expected_mask.shape == (1,):
            expected_mask = expected_mask[:, np.newaxis]
        c = int(expected_mask.shape[0] / 2)
        assert_equal(expected_mask, actual_mask[c, :, :])
        assert_equal(expected_mask, actual_mask[:, c, :])
        assert_equal(expected_mask, actual_mask[:, :, c])
        k = k + 1