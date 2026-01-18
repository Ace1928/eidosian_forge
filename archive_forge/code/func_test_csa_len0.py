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
def test_csa_len0():
    csa_info = csa.read(CSA2_0len)
    assert csa_info['type'] == 2
    assert csa_info['n_tags'] == 44
    tags = csa_info['tags']
    assert len(tags) == 44