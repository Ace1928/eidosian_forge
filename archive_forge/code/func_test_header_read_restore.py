import copy
import os
import sys
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arr_dict_equal, clear_and_catch_warnings, data_path, error_warnings
from .. import trk as trk_module
from ..header import Field
from ..tractogram import Tractogram
from ..tractogram_file import HeaderError, HeaderWarning
from ..trk import (
from .test_tractogram import assert_tractogram_equal
def test_header_read_restore(self):
    trk_fname = DATA['simple_trk_fname']
    bio = BytesIO()
    bio.write(b'Along my very merry way')
    hdr_pos = bio.tell()
    hdr_from_fname = TrkFile._read_header(trk_fname)
    with open(trk_fname, 'rb') as fobj:
        bio.write(fobj.read())
    bio.seek(hdr_pos)
    hdr_from_fname['_offset_data'] += hdr_pos
    assert_arr_dict_equal(TrkFile._read_header(bio), hdr_from_fname)
    assert bio.tell() == hdr_pos