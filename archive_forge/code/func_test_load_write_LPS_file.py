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
def test_load_write_LPS_file(self):
    trk_RAS = TrkFile.load(DATA['standard_trk_fname'], lazy_load=False)
    trk_LPS = TrkFile.load(DATA['standard_LPS_trk_fname'], lazy_load=False)
    assert_tractogram_equal(trk_LPS.tractogram, trk_RAS.tractogram)
    trk_file = BytesIO()
    trk = TrkFile(trk_LPS.tractogram, trk_LPS.header)
    trk.save(trk_file)
    trk_file.seek(0, os.SEEK_SET)
    new_trk = TrkFile.load(trk_file)
    assert_arr_dict_equal(new_trk.header, trk.header)
    assert_tractogram_equal(new_trk.tractogram, trk.tractogram)
    new_trk_orig = TrkFile.load(DATA['standard_LPS_trk_fname'])
    assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
    trk_file.seek(0, os.SEEK_SET)
    assert trk_file.read() == open(DATA['standard_LPS_trk_fname'], 'rb').read()
    trk_file = BytesIO()
    header = copy.deepcopy(trk_LPS.header)
    header[Field.VOXEL_ORDER] = b''
    trk = TrkFile(trk_LPS.tractogram, header)
    trk.save(trk_file)
    trk_file.seek(0, os.SEEK_SET)
    new_trk = TrkFile.load(trk_file)
    assert_arr_dict_equal(new_trk.header, trk_LPS.header)
    assert_tractogram_equal(new_trk.tractogram, trk.tractogram)
    new_trk_orig = TrkFile.load(DATA['standard_LPS_trk_fname'])
    assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
    trk_file.seek(0, os.SEEK_SET)
    assert trk_file.read() == open(DATA['standard_LPS_trk_fname'], 'rb').read()