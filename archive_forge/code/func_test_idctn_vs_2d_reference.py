from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('funcn,func', [(idctn, idct), (idstn, idst)])
@pytest.mark.parametrize('dct_type', dct_type)
@pytest.mark.parametrize('norm', norms)
def test_idctn_vs_2d_reference(self, funcn, func, dct_type, norm):
    fdata = dctn(self.data, type=dct_type, norm=norm)
    y1 = funcn(fdata, type=dct_type, norm=norm)
    y2 = ref_2d(func, fdata, type=dct_type, norm=norm)
    assert_array_almost_equal(y1, y2, decimal=11)