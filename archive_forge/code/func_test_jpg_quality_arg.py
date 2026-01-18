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
def test_jpg_quality_arg():
    chessboard = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    with temporary_file(suffix='.jpg') as jpg:
        imsave(jpg, chessboard, quality=95)
        im = imread(jpg)
        sim = structural_similarity(chessboard, im, data_range=chessboard.max() - chessboard.min())
        assert sim > 0.99