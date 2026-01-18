import os
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import data_path, error_warnings
from .. import tck as tck_module
from ..array_sequence import ArraySequence
from ..tck import TckFile
from ..tractogram import Tractogram
from ..tractogram_file import DataError, HeaderError, HeaderWarning
from .test_tractogram import assert_tractogram_equal
def test_load_simple_file_in_big_endian(self):
    for lazy_load in [False, True]:
        tck = TckFile.load(DATA['simple_tck_big_endian_fname'], lazy_load=lazy_load)
        with pytest.warns(Warning) if lazy_load else error_warnings():
            assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])
        assert tck.header['datatype'] == 'Float32BE'