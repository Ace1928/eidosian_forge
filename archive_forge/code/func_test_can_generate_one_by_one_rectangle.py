import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_can_generate_one_by_one_rectangle():
    image, labels = random_shapes((50, 128), max_shapes=1, min_size=1, max_size=1, shape='rectangle')
    assert len(labels) == 1
    _, bbox = labels[0]
    crop = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    assert np.shape(crop) == (1, 1, 3) and np.any(crop >= 1) and np.any(crop < 255)