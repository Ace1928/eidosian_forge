import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
def test_arraymap_call():
    ar = np.array([1, 1, 5, 5, 8, 99, 42, 0], dtype=np.intp)
    relabeled, fw, inv = relabel_sequential(ar)
    testing.assert_array_equal(relabeled, fw(ar))
    testing.assert_array_equal(ar, inv(relabeled))