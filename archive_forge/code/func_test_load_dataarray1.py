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
def test_load_dataarray1():
    img1 = load(DATA_FILE1)
    with InTemporaryDirectory():
        save(img1, 'test.gii')
        bimg = load('test.gii')
    for img in (img1, bimg):
        assert_array_almost_equal(img.darrays[0].data, DATA_FILE1_darr1)
        assert_array_almost_equal(img.darrays[1].data, DATA_FILE1_darr2)
        me = img.darrays[0].meta
        assert 'AnatomicalStructurePrimary' in me
        assert 'AnatomicalStructureSecondary' in me
        me['AnatomicalStructurePrimary'] == 'CortexLeft'
        assert_array_almost_equal(img.darrays[0].coordsys.xform, np.eye(4, 4))
        assert xform_codes.niistring[img.darrays[0].coordsys.dataspace] == 'NIFTI_XFORM_TALAIRACH'
        assert xform_codes.niistring[img.darrays[0].coordsys.xformspace] == 'NIFTI_XFORM_TALAIRACH'