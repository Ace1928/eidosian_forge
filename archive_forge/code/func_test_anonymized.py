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
def test_anonymized():
    with open(ANON_PAR) as fobj:
        anon_hdr = PARRECHeader.from_fileobj(fobj)
    gen_defs, img_defs = (anon_hdr.general_info, anon_hdr.image_defs)
    assert gen_defs['patient_name'] == ''
    assert gen_defs['exam_name'] == ''
    assert gen_defs['protocol_name'] == ''
    assert gen_defs['series_type'] == 'Image   MRSERIES'
    assert_almost_equal(img_defs['window center'][0], -2374.72283272283, 6)
    assert_almost_equal(img_defs['window center'][-1], 236.385836385836, 6)
    assert_almost_equal(img_defs['window width'][0], 767.277167277167, 6)
    assert_almost_equal(img_defs['window width'][-1], 236.385836385836, 6)