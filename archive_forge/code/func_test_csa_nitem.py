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
def test_csa_nitem():
    with pytest.raises(csa.CSAReadError):
        csa.read(CSA_STR_1001n_items)
    csa_info = csa.read(CSA_STR_valid)
    assert len(csa_info['tags']) == 1
    n_items_thresh = csa.MAX_CSA_ITEMS
    try:
        csa.MAX_CSA_ITEMS = 2000
        csa_info = csa.read(CSA_STR_1001n_items)
        assert len(csa_info['tags']) == 1
    finally:
        csa.MAX_CSA_ITEMS = n_items_thresh