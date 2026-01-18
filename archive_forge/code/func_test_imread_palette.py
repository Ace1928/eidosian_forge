from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
from skimage.io import imread, imsave, use_plugin, reset_plugins
from skimage._shared import testing
from skimage._shared.testing import (
from pytest import importorskip
importorskip('imread')
def test_imread_palette():
    img = imread(fetch('data/palette_color.png'))
    assert img.ndim == 3