import os
import pathlib
import tempfile
import numpy as np
import pytest
from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir
def test_imread_file_url():
    data_path = str(fetch('data/camera.png'))
    data_path = data_path.replace(os.path.sep, '/')
    image_url = f'file:///{data_path}'
    image = io.imread(image_url)
    assert image.shape == (512, 512)