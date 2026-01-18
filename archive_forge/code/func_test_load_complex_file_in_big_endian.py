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
def test_load_complex_file_in_big_endian(self):
    trk_struct, trk_bytes = self.trk_with_bytes('complex_trk_big_endian_fname', endian='>')
    good_orders = '>' if sys.byteorder == 'little' else '>='
    hdr_size = trk_struct['hdr_size']
    assert hdr_size.dtype.byteorder in good_orders
    assert hdr_size == 1000
    for lazy_load in [False, True]:
        trk = TrkFile.load(DATA['complex_trk_big_endian_fname'], lazy_load=lazy_load)
        with pytest.warns(Warning) if lazy_load else error_warnings():
            assert_tractogram_equal(trk.tractogram, DATA['complex_tractogram'])