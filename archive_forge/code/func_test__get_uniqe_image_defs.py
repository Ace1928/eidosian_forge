from glob import glob
from os.path import basename, dirname
from os.path import join as pjoin
from warnings import simplefilter
import numpy as np
import pytest
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import load as top_load
from .. import parrec
from ..fileholders import FileHolder
from ..nifti1 import Nifti1Extension, Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..parrec import (
from ..testing import assert_arr_dict_equal, clear_and_catch_warnings, suppress_warnings
from ..volumeutils import array_from_file
from . import test_spatialimages as tsi
from .test_arrayproxy import check_mmap
def test__get_uniqe_image_defs():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS.copy())
    uip = hdr._get_unique_image_prop
    assert uip('image pixel size') == 16
    hdr.image_defs['image pixel size'][3] = 32
    with pytest.raises(PARRECError):
        uip('image pixel size')
    assert_array_equal(uip('recon resolution'), [64, 64])
    hdr.image_defs['recon resolution'][4, 1] = 32
    with pytest.raises(PARRECError):
        uip('recon resolution')
    assert_array_equal(uip('image angulation'), [-13.26, 0, 0])
    hdr.image_defs['image angulation'][5, 2] = 1
    with pytest.raises(PARRECError):
        uip('image angulation')
    with pytest.raises(PARRECError):
        uip('slice number')