import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_generates_white_image_when_intensity_range_255():
    image, labels = random_shapes((128, 128), max_shapes=3, intensity_range=((255, 255),), rng=42)
    assert len(labels) > 0
    assert (image == 255).all()