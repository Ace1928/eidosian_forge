import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_generates_color_images_with_correct_shape():
    image, _ = random_shapes((128, 128), max_shapes=10)
    assert image.shape == (128, 128, 3)