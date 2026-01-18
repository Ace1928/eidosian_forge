from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'uint8', 'int'])
def test_dtype_support(dtype):
    img = np.random.rand(28, 28).astype(dtype)
    slic(img, start_label=1, channel_axis=None)