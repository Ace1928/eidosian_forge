import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_generate_triangle_throws_when_size_too_small():
    with testing.raises(ValueError):
        random_shapes((128, 64), max_shapes=1, min_size=1, max_size=1, shape='triangle')