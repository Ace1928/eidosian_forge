import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
def test_join_segmentations():
    s1 = np.array([[0, 0, 1, 1], [0, 2, 1, 1], [2, 2, 2, 1]])
    s2 = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1]])
    j = join_segmentations(s1, s2)
    j_ref = np.array([[0, 1, 3, 2], [0, 5, 3, 2], [4, 5, 5, 3]])
    assert_array_equal(j, j_ref)
    j, m1, m2 = join_segmentations(s1, s2, return_mapping=True)
    assert_array_equal(m1[j], s1)
    assert_array_equal(m2[j], s2)
    s3 = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
    with testing.raises(ValueError):
        join_segmentations(s1, s3)