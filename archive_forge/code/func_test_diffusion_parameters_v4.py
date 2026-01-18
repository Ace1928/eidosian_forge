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
def test_diffusion_parameters_v4():
    dti_v4_par = pjoin(DATA_PATH, 'DTIv40.PAR')
    with open(dti_v4_par) as fobj:
        dti_v4_hdr = PARRECHeader.from_fileobj(fobj)
    assert dti_v4_hdr.get_data_shape() == (80, 80, 10, 8)
    assert dti_v4_hdr.general_info['diffusion'] == 1
    bvals, bvecs = dti_v4_hdr.get_bvals_bvecs()
    assert_almost_equal(bvals, DTI_PAR_BVALS)
    assert bvecs is None
    assert dti_v4_hdr.get_q_vectors() is None