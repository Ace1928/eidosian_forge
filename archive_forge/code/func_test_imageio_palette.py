from tempfile import NamedTemporaryFile
import numpy as np
from skimage.io import imread, imsave, plugin_order
from skimage._shared import testing
from skimage._shared.testing import fetch, assert_stacklevel
import pytest
def test_imageio_palette():
    img = imread(fetch('data/palette_color.png'))
    assert img.ndim == 3