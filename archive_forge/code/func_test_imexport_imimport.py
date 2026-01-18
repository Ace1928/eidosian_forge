import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
from PIL import Image
from skimage._shared import testing
from skimage._shared._tempfile import temporary_file
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.metrics import structural_similarity
from ... import img_as_float
from ...color import rgb2lab
from .. import imread, imsave, reset_plugins, use_plugin, plugin_order
from .._plugins.pil_plugin import _palette_is_grayscale, ndarray_to_pil, pil_to_ndarray
def test_imexport_imimport():
    shape = (2, 2)
    image = np.zeros(shape)
    pil_image = ndarray_to_pil(image)
    out = pil_to_ndarray(pil_image)
    assert_equal(out.shape, shape)