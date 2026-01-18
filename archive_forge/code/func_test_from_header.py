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
def test_from_header(self):
    klass = self.header_class
    empty = klass.from_header()
    assert klass() == empty
    empty = klass.from_header(None)
    assert klass() == empty
    hdr = klass()
    hdr.set_data_dtype(np.float64)
    hdr.set_data_shape((1, 2, 3))
    hdr.set_zooms((3.0, 2.0, 1.0))
    for check in (True, False):
        copy = klass.from_header(hdr, check=check)
        assert hdr == copy
        assert hdr is not copy

    class C:

        def get_data_dtype(self):
            return np.dtype('i2')

        def get_data_shape(self):
            return (5, 4, 3)

        def get_zooms(self):
            return (10.0, 9.0, 8.0)
    converted = klass.from_header(C())
    assert isinstance(converted, klass)
    assert converted.get_data_dtype() == np.dtype('i2')
    assert converted.get_data_shape() == (5, 4, 3)
    assert converted.get_zooms() == (10.0, 9.0, 8.0)