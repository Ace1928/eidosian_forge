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
def test_dualTR():
    expected_TRs = np.asarray([2000.0, 500.0])
    with open(DUAL_TR_PAR) as fobj:
        with clear_and_catch_warnings(modules=[parrec], record=True) as wlist:
            simplefilter('always')
            dualTR_hdr = PARRECHeader.from_fileobj(fobj)
        assert len(wlist) == 1
        assert_array_equal(dualTR_hdr.general_info['repetition_time'], expected_TRs)
        assert dualTR_hdr.get_zooms()[3] == expected_TRs[0] / 1000