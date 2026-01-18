import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_generates_gray_images_with_correct_shape():
    image, _ = random_shapes((4567, 123), min_shapes=3, max_shapes=20, channel_axis=None)
    assert image.shape == (4567, 123)