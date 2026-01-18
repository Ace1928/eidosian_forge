import os
import warnings
from pathlib import Path
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..ecat import (
from ..openers import Opener
from ..testing import data_path, suppress_warnings
from ..tmpdirs import InTemporaryDirectory
from . import test_wrapstruct as tws
from .test_fileslice import slicer_samples
def test_mlist(self):
    fid = open(self.example_file, 'rb')
    hdr = self.header_class.from_fileobj(fid)
    mlist = read_mlist(fid, hdr.endianness)
    fid.seek(0)
    fid.seek(512)
    dat = fid.read(128 * 32)
    dt = np.dtype([('matlist', np.int32)])
    dt = dt.newbyteorder('>')
    mats = np.recarray(shape=(32, 4), dtype=dt, buf=dat)
    fid.close()
    assert mats['matlist'][0, 0] + mats['matlist'][0, 3] == 31
    assert get_frame_order(mlist)[0][0] == 0
    assert get_frame_order(mlist)[0][1] == 16842758.0
    badordermlist = np.array([[16842754.0, 3.0, 12035.0, 1.0], [16842753.0, 12036.0, 24068.0, 1.0], [16842755.0, 24069.0, 36101.0, 1.0], [16842756.0, 36102.0, 48134.0, 1.0], [16842757.0, 48135.0, 60167.0, 1.0], [16842758.0, 60168.0, 72200.0, 1.0]])
    with suppress_warnings():
        assert get_frame_order(badordermlist)[0][0] == 1