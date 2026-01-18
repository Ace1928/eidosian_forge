from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
@pytest.mark.xfail(Version(np.__version__) >= Version('2.0.0.dev0'), reason='tifffile uses deprecated attribute `ndarray.newbyteorder`')
def test_kidney_3d_multichannel():
    """Test that 3D multichannel image of kidney tissue can be loaded.

    Needs internet connection.
    """
    fetch('data/kidney.tif')
    kidney = data.kidney()
    assert kidney.shape == (16, 512, 512, 3)