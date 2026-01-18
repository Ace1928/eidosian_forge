from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
def test_data_shape(self):
    img_klass = self.image_class
    arr = np.arange(4, dtype=np.int16)
    img = img_klass(arr, np.eye(4))
    assert img.shape[:1] == (4,)
    assert np.prod(img.shape) == 4
    img = img_klass(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
    assert img.shape == (2, 3, 4)