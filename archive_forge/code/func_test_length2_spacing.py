import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_length2_spacing():
    np.random.seed(42)
    img = np.ones((10, 10)) + 0.2 * np.random.normal(size=(10, 10))
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[2, 4] = 1
    labels[6, 8] = 4
    random_walker(img, labels, spacing=(1.0, 2.0))