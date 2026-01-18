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
def test_diffusion_parameters_strict_sort():
    dti_par = pjoin(DATA_PATH, 'DTI.PAR')
    with open(dti_par) as fobj:
        dti_hdr = PARRECHeader.from_fileobj(fobj, strict_sort=True)
    np.random.shuffle(dti_hdr.image_defs)
    assert dti_hdr.get_data_shape() == (80, 80, 10, 8)
    assert dti_hdr.general_info['diffusion'] == 1
    bvals, bvecs = dti_hdr.get_bvals_bvecs()
    assert_almost_equal(bvals, np.sort(DTI_PAR_BVALS))
    assert_almost_equal(bvecs, DTI_PAR_BVECS[np.ix_(np.argsort(DTI_PAR_BVALS, kind='stable'), [2, 0, 1])])
    assert_almost_equal(dti_hdr.get_q_vectors(), bvals[:, None] * bvecs)