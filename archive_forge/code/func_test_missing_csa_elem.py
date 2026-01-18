import gzip
import sys
from copy import deepcopy
from os.path import join as pjoin
import numpy as np
import pytest
from .. import csareader as csa
from .. import dwiparams as dwp
from . import dicom_test, pydicom
from .test_dicomwrappers import DATA, IO_DATA_PATH
@dicom_test
def test_missing_csa_elem():
    dcm = deepcopy(DATA)
    csa_tag = pydicom.dataset.Tag(41, 4112)
    del dcm[csa_tag]
    hdr = csa.get_csa_header(dcm, 'image')
    assert hdr is None