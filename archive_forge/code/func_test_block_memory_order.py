import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_block_memory_order(self, block):
    arr_c = np.zeros((3,) * 3, order='C')
    arr_f = np.zeros((3,) * 3, order='F')
    b_c = [[[arr_c, arr_c], [arr_c, arr_c]], [[arr_c, arr_c], [arr_c, arr_c]]]
    b_f = [[[arr_f, arr_f], [arr_f, arr_f]], [[arr_f, arr_f], [arr_f, arr_f]]]
    assert block(b_c).flags['C_CONTIGUOUS']
    assert block(b_f).flags['F_CONTIGUOUS']
    arr_c = np.zeros((3, 3), order='C')
    arr_f = np.zeros((3, 3), order='F')
    b_c = [[arr_c, arr_c], [arr_c, arr_c]]
    b_f = [[arr_f, arr_f], [arr_f, arr_f]]
    assert block(b_c).flags['C_CONTIGUOUS']
    assert block(b_f).flags['F_CONTIGUOUS']