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
def test_image_creation():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    arr_prox_dv = np.array(PARRECArrayProxy(EG_REC, hdr, scaling='dv'))
    arr_prox_fp = np.array(PARRECArrayProxy(EG_REC, hdr, scaling='fp'))
    good_map = dict(image=FileHolder(EG_REC), header=FileHolder(EG_PAR))
    trunc_map = dict(image=FileHolder(TRUNC_REC), header=FileHolder(TRUNC_PAR))
    for func, good_param, trunc_param in ((PARRECImage.from_filename, EG_PAR, TRUNC_PAR), (PARRECImage.load, EG_PAR, TRUNC_PAR), (parrec.load, EG_PAR, TRUNC_PAR), (PARRECImage.from_file_map, good_map, trunc_map)):
        img = func(good_param)
        assert_array_equal(img.dataobj, arr_prox_dv)
        with pytest.raises(TypeError):
            func(good_param, False)
        img = func(good_param, permit_truncated=False)
        assert_array_equal(img.dataobj, arr_prox_dv)
        with pytest.raises(TypeError):
            func(good_param, False, 'dv')
        img = func(good_param, permit_truncated=False, scaling='dv')
        assert_array_equal(img.dataobj, arr_prox_dv)
        img = func(good_param, scaling='dv')
        assert_array_equal(img.dataobj, arr_prox_dv)
        img = func(good_param, scaling='fp')
        assert_array_equal(img.dataobj, arr_prox_fp)
        with pytest.raises(PARRECError):
            func(trunc_param)
        with pytest.raises(PARRECError):
            func(trunc_param, permit_truncated=False)
        with pytest.warns(UserWarning, match='Header inconsistency'):
            img = func(trunc_param, permit_truncated=True)
        assert_array_equal(img.dataobj, arr_prox_dv)
        with pytest.warns(UserWarning, match='Header inconsistency'):
            img = func(trunc_param, permit_truncated=True, scaling='dv')
        assert_array_equal(img.dataobj, arr_prox_dv)
        with pytest.warns(UserWarning, match='Header inconsistency'):
            img = func(trunc_param, permit_truncated=True, scaling='fp')
        assert_array_equal(img.dataobj, arr_prox_fp)