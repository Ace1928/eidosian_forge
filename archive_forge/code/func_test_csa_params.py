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
def test_csa_params():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        n_o_m = csa.get_n_mosaic(csa_info)
        assert n_o_m == 48
        snv = csa.get_slice_normal(csa_info)
        assert snv.shape == (3,)
        assert np.allclose(1, np.sqrt((snv * snv).sum()))
        amt = csa.get_acq_mat_txt(csa_info)
        assert amt == '128p*128'
    csa_info = csa.read(CSA2_B0)
    b_matrix = csa.get_b_matrix(csa_info)
    assert b_matrix is None
    b_value = csa.get_b_value(csa_info)
    assert b_value == 0
    g_vector = csa.get_g_vector(csa_info)
    assert g_vector is None
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa.get_b_matrix(csa_info)
    assert b_matrix.shape == (3, 3)
    dwp.B2q(b_matrix)
    b_value = csa.get_b_value(csa_info)
    assert b_value == 1000
    g_vector = csa.get_g_vector(csa_info)
    assert g_vector.shape == (3,)
    assert np.allclose(1, np.sqrt((g_vector * g_vector).sum()))