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
def test_ice_dims():
    ex_dims0 = ['X', '1', '1', '1', '1', '1', '1', '48', '1', '1', '1', '1', '201']
    ex_dims1 = ['X', '1', '1', '1', '2', '1', '1', '48', '1', '1', '1', '1', '201']
    for csa_str, ex_dims in ((CSA2_B0, ex_dims0), (CSA2_B1000, ex_dims1)):
        csa_info = csa.read(csa_str)
        assert csa.get_ice_dims(csa_info) == ex_dims
    assert csa.get_ice_dims({}) is None