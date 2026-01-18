from contextlib import contextmanager
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage import io
from skimage.io import manage_plugins
@contextmanager
def protect_preferred_plugins():
    """Contexts where `preferred_plugins` can be modified w/o side-effects."""
    preferred_plugins = manage_plugins.preferred_plugins.copy()
    try:
        yield
    finally:
        manage_plugins.preferred_plugins = preferred_plugins