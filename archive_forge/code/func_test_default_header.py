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
def test_default_header(self):
    arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    img = self.image_class(arr, None)
    hdr = self.image_class.header_class()
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    hdr.set_data_offset(0)
    hdr.set_slope_inter(np.nan, np.nan)
    assert img.header == hdr