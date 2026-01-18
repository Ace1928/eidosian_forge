import io
import os
import pathlib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ... import imageglobals
from ...fileholders import FileHolder
from ...openers import ImageOpener
from ...spatialimages import HeaderDataError
from ...testing import data_path
from ...tests import test_spatialimages as tsi
from ...tests import test_wrapstruct as tws
from ...tmpdirs import InTemporaryDirectory
from ...volumeutils import sys_is_le
from ...wrapstruct import WrapStructError
from .. import load, save
from ..mghformat import MGHError, MGHHeader, MGHImage
def test_mgh_set_data_shape():
    hdr = MGHHeader()
    hdr.set_data_shape((5,))
    assert_array_equal(hdr.get_data_shape(), (5, 1, 1))
    hdr.set_data_shape((5, 4))
    assert_array_equal(hdr.get_data_shape(), (5, 4, 1))
    hdr.set_data_shape((5, 4, 3))
    assert_array_equal(hdr.get_data_shape(), (5, 4, 3))
    hdr.set_data_shape((5, 4, 3, 2))
    assert_array_equal(hdr.get_data_shape(), (5, 4, 3, 2))
    with pytest.raises(ValueError):
        hdr.set_data_shape((5, 4, 3, 2, 1))