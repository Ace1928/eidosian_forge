import os
import itertools
import numpy as np
import imageio.v3 as iio3
from skimage import data_dir
from skimage.io.collection import ImageCollection, MultiImage, alphanumeric_key
from skimage.io import reset_plugins
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose, fetch
import pytest
@pytest.mark.skipif(not has_pooch, reason='needs pooch to download data')
def test_custom_load_func_sequence(self):
    filename = fetch('data/no_time_for_that_tiny.gif')

    def reader(index):
        return iio3.imread(filename, index=index)
    ic = ImageCollection(range(24), load_func=reader)
    assert len(ic) == 24
    assert ic[0].shape == (25, 14, 3)