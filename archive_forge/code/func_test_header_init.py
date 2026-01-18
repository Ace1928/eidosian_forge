from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
def test_header_init():
    hdr = SpatialHeader()
    assert hdr.get_data_dtype() == np.dtype(np.float32)
    assert hdr.get_data_shape() == (0,)
    assert hdr.get_zooms() == (1.0,)
    hdr = SpatialHeader(np.float64)
    assert hdr.get_data_dtype() == np.dtype(np.float64)
    assert hdr.get_data_shape() == (0,)
    assert hdr.get_zooms() == (1.0,)
    hdr = SpatialHeader(np.float64, shape=(1, 2, 3))
    assert hdr.get_data_dtype() == np.dtype(np.float64)
    assert hdr.get_data_shape() == (1, 2, 3)
    assert hdr.get_zooms() == (1.0, 1.0, 1.0)
    hdr = SpatialHeader(np.float64, shape=(1, 2, 3), zooms=None)
    assert hdr.get_data_dtype() == np.dtype(np.float64)
    assert hdr.get_data_shape() == (1, 2, 3)
    assert hdr.get_zooms() == (1.0, 1.0, 1.0)
    hdr = SpatialHeader(np.float64, shape=(1, 2, 3), zooms=(3.0, 2.0, 1.0))
    assert hdr.get_data_dtype() == np.dtype(np.float64)
    assert hdr.get_data_shape() == (1, 2, 3)
    assert hdr.get_zooms() == (3.0, 2.0, 1.0)