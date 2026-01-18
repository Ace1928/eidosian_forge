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
def test_epi_params():
    for par_root in ('T2_-interleaved', 'T2_', 'phantom_EPI_asc_CLEAR_2_1'):
        epi_par = pjoin(DATA_PATH, par_root + '.PAR')
        with open(epi_par) as fobj:
            epi_hdr = PARRECHeader.from_fileobj(fobj)
        assert len(epi_hdr.get_data_shape()) == 4
        assert_almost_equal(epi_hdr.get_zooms()[-1], 2.0)