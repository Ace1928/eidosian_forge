import itertools
import logging
import os
import pickle
import re
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import imageglobals
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..arraywriters import WriterError
from ..casting import sctypes_aliases
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spatialimages import HeaderDataError, HeaderTypeError, supported_np_types
from ..testing import (
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from . import test_wrapstruct as tws
def test_data_hdr_cache(self):
    IC = self.image_class
    fm = IC.make_file_map()
    for key, value in fm.items():
        fm[key].fileobj = BytesIO()
    shape = (2, 3, 4)
    data = np.arange(24, dtype=np.int8).reshape(shape)
    affine = np.eye(4)
    hdr = IC.header_class()
    hdr.set_data_dtype(np.int16)
    img = IC(data, affine, hdr)
    img.to_file_map(fm)
    img2 = IC.from_file_map(fm)
    assert img2.shape == shape
    assert img2.get_data_dtype().type == np.int16
    hdr = img2.header
    hdr.set_data_shape((3, 2, 2))
    assert hdr.get_data_shape() == (3, 2, 2)
    hdr.set_data_dtype(np.uint8)
    assert hdr.get_data_dtype() == np.dtype(np.uint8)
    assert_array_equal(img2.get_fdata(), data)
    assert_array_equal(np.asanyarray(img2.dataobj), data)