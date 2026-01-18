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
def test_as_byteswapped(self):
    hdr = self.header_class()
    assert hdr.endianness == '>'
    for endianness in BIG_CODES:
        hdr2 = hdr.as_byteswapped(endianness)
        assert hdr2 is not hdr
        assert hdr2 == hdr
    for endianness in (None,) + LITTLE_CODES:
        with pytest.raises(ValueError):
            hdr.as_byteswapped(endianness)

    class DC(self.header_class):

        def check_fix(self, *args, **kwargs):
            raise Exception
    with pytest.raises(Exception):
        DC(hdr.binaryblock)
    hdr = DC(hdr.binaryblock, check=False)
    hdr2 = hdr.as_byteswapped('>')