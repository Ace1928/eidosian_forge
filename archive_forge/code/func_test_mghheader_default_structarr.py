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
def test_mghheader_default_structarr():
    hdr = MGHHeader.default_structarr()
    assert hdr['version'] == 1
    assert_array_equal(hdr['dims'], 1)
    assert hdr['type'] == 3
    assert hdr['dof'] == 0
    assert hdr['goodRASFlag'] == 1
    assert_array_equal(hdr['delta'], 1)
    assert_array_equal(hdr['Mdc'], [[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert_array_equal(hdr['Pxyz_c'], 0)
    assert hdr['tr'] == 0
    assert hdr['flip_angle'] == 0
    assert hdr['te'] == 0
    assert hdr['ti'] == 0
    assert hdr['fov'] == 0
    for endianness in (None,) + BIG_CODES:
        hdr2 = MGHHeader.default_structarr(endianness=endianness)
        assert hdr2 == hdr
        assert hdr2.view(hdr2.dtype.newbyteorder('>')) == hdr
    for endianness in LITTLE_CODES:
        with pytest.raises(ValueError):
            MGHHeader.default_structarr(endianness=endianness)