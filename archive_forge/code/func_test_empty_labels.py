import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_empty_labels():
    image = np.random.random((5, 5))
    labels = np.zeros((5, 5), dtype=int)
    with testing.raises(ValueError, match='No seeds provided'):
        random_walker(image, labels)
    labels[1, 1] = -1
    with testing.raises(ValueError, match='No seeds provided'):
        random_walker(image, labels)
    labels[3, 3] = 1
    random_walker(image, labels)