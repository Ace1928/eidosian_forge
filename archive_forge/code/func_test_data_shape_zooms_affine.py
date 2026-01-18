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
def test_data_shape_zooms_affine(self):
    hdr = self.header_class()
    for shape in ((1, 2, 3), (0,), (1,), (1, 2), (1, 2, 3, 4)):
        L = len(shape)
        hdr.set_data_shape(shape)
        if L:
            assert hdr.get_data_shape() == shape
        else:
            assert hdr.get_data_shape() == (0,)
        assert hdr.get_zooms() == (1,) * L
        if len(shape):
            with pytest.raises(HeaderDataError):
                hdr.set_zooms((1,) * (L - 1))
            with pytest.raises(HeaderDataError):
                hdr.set_zooms((-1,) + (1,) * (L - 1))
        with pytest.raises(HeaderDataError):
            hdr.set_zooms((1,) * (L + 1))
        with pytest.raises(HeaderDataError):
            hdr.set_zooms((-1,) * L)
    hdr = self.header_class()
    hdr.set_data_shape((1, 2, 3))
    hdr.set_zooms((4, 5, 6))
    assert_array_equal(hdr.get_zooms(), (4, 5, 6))
    hdr.set_data_shape((1, 2))
    assert_array_equal(hdr.get_zooms(), (4, 5))
    hdr.set_data_shape((1, 2, 3))
    assert_array_equal(hdr.get_zooms(), (4, 5, 1))
    assert_array_equal(np.diag(hdr.get_base_affine()), [-4, 5, 1, 1])
    hdr.set_zooms((1, 1, 1))
    assert_array_equal(np.diag(hdr.get_base_affine()), [-1, 1, 1, 1])