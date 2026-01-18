from tempfile import NamedTemporaryFile
import numpy as np
from skimage.io import imread, imsave, plugin_order
from skimage._shared import testing
from skimage._shared.testing import fetch, assert_stacklevel
import pytest
def test_imageio_truncated_jpg():
    with testing.raises((OSError, SyntaxError)):
        imread(fetch('data/truncated.jpg'))