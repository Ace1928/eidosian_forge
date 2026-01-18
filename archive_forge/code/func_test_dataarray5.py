import shutil
import sys
import warnings
from os.path import basename, dirname
from os.path import join as pjoin
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from ...loadsave import load, save
from ...nifti1 import xform_codes
from ...testing import clear_and_catch_warnings, suppress_warnings
from ...tmpdirs import InTemporaryDirectory
from .. import gifti as gi
from ..parse_gifti_fast import GiftiImageParser, GiftiParseError
from ..util import gifti_endian_codes
def test_dataarray5():
    img5 = load(DATA_FILE5)
    for da in img5.darrays:
        gifti_endian_codes.byteorder[da.endian] == 'little'
    assert_array_almost_equal(img5.darrays[0].data, DATA_FILE5_darr1)
    assert_array_almost_equal(img5.darrays[1].data, DATA_FILE5_darr2)