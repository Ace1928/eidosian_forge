import numpy as np
from skimage._shared.testing import assert_equal
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import BRIEF, match_descriptors, corner_peaks, corner_harris
from skimage._shared import testing
def test_max_ratio():
    descs1 = 10 * np.arange(10)[:, None].astype(np.float32)
    descs2 = 10 * np.arange(15)[:, None].astype(np.float32)
    descs2[0] = 5.0
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=1.0, cross_check=False)
    assert_equal(len(matches), 10)
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=0.6, cross_check=False)
    assert_equal(len(matches), 10)
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=0.5, cross_check=False)
    assert_equal(len(matches), 9)
    descs1[0] = 7.5
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=0.5, cross_check=False)
    assert_equal(len(matches), 9)
    descs2 = 10 * np.arange(1)[:, None].astype(np.float32)
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=1.0, cross_check=False)
    assert_equal(len(matches), 10)
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=0.5, cross_check=False)
    assert_equal(len(matches), 10)
    descs1 = 10 * np.arange(1)[:, None].astype(np.float32)
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=1.0, cross_check=False)
    assert_equal(len(matches), 1)
    matches = match_descriptors(descs1, descs2, metric='euclidean', max_ratio=0.5, cross_check=False)
    assert_equal(len(matches), 1)