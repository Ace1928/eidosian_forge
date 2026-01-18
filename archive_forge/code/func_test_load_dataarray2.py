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
def test_load_dataarray2():
    img2 = load(DATA_FILE2)
    with InTemporaryDirectory():
        save(img2, 'test.gii')
        bimg = load('test.gii')
    for img in (img2, bimg):
        assert_array_almost_equal(img.darrays[0].data[:10], DATA_FILE2_darr1)