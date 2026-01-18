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
def test_alternative_header_field_names():
    with ImageOpener(VARIANT_PAR, 'rt') as _fobj:
        HDR_INFO, HDR_DEFS = parse_PAR_header(_fobj)
    assert HDR_INFO['series_type'] == 'Image   MRSERIES'
    assert HDR_INFO['diffusion_echo_time'] == 0.0
    assert HDR_INFO['repetition_time'] == npa([21225.76])
    assert HDR_INFO['patient_position'] == 'HFS'