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
def test_custom_load_func_w_kwarg(self):
    load_pattern = fetch('data/no_time_for_that_tiny.gif')

    def load_fn(f, step):
        vid = iio3.imiter(f)
        return list(itertools.islice(vid, None, None, step))
    ic = ImageCollection(load_pattern, load_func=load_fn, step=3)
    assert len(ic) == 1
    assert len(ic[0]) == 8