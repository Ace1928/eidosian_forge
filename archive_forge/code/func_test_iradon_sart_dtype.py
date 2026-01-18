import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_iradon_sart_dtype(dtype):
    sinogram = np.zeros((16, 1), dtype=int)
    sinogram[8, 0] = 1.0
    sinogram64 = sinogram.astype('float64')
    sinogram32 = sinogram.astype('float32')
    with expected_warnings(['Input data is cast to float']):
        assert iradon_sart(sinogram, theta=[0]).dtype == 'float64'
    assert iradon_sart(sinogram64, theta=[0]).dtype == sinogram64.dtype
    assert iradon_sart(sinogram32, theta=[0]).dtype == sinogram32.dtype
    assert iradon_sart(sinogram, theta=[0], dtype=dtype).dtype == dtype
    assert iradon_sart(sinogram32, theta=[0], dtype=dtype).dtype == dtype
    assert iradon_sart(sinogram64, theta=[0], dtype=dtype).dtype == dtype