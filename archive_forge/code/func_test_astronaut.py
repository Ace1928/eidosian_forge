from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
def test_astronaut():
    """Test that "astronaut" image can be loaded."""
    astronaut = data.astronaut()
    assert_equal(astronaut.shape, (512, 512, 3))