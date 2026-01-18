from contextlib import contextmanager
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage import io
from skimage.io import manage_plugins
def test_failed_use():
    with pytest.raises(ValueError):
        manage_plugins.use_plugin('asd')