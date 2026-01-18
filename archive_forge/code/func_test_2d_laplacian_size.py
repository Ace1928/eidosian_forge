import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_2d_laplacian_size():
    data = np.asarray([[12823, 12787, 12710], [12883, 13425, 12067], [11934, 11929, 12309]])
    markers = np.asarray([[0, -1, 2], [0, -1, 0], [1, 0, -1]])
    expected_labels = np.asarray([[1, -1, 2], [1, -1, 2], [1, 1, -1]])
    labels = random_walker(data, markers, beta=10)
    np.testing.assert_array_equal(labels, expected_labels)