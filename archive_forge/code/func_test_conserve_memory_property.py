import os
import numpy as np
from skimage.io import use_plugin, reset_plugins
from skimage.io.collection import MultiImage
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose
from pytest import fixture
def test_conserve_memory_property(imgs):
    for img in imgs:
        assert isinstance(img.conserve_memory, bool)
        with testing.raises(AttributeError):
            img.conserve_memory = True