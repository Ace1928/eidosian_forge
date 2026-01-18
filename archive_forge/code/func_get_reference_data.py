from os.path import join, dirname
from typing import Callable, Dict, Tuple, Union, Type
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
def get_reference_data():
    ref = getattr(globals(), '__reference_data', None)
    if ref is not None:
        return ref
    MDATA = np.load(join(fftpack_test_dir, 'test.npz'))
    X = [MDATA['x%d' % i] for i in range(MDATA_COUNT)]
    Y = [MDATA['y%d' % i] for i in range(MDATA_COUNT)]
    FFTWDATA_DOUBLE = np.load(join(fftpack_test_dir, 'fftw_double_ref.npz'))
    FFTWDATA_SINGLE = np.load(join(fftpack_test_dir, 'fftw_single_ref.npz'))
    FFTWDATA_SIZES = FFTWDATA_DOUBLE['sizes']
    assert len(FFTWDATA_SIZES) == FFTWDATA_COUNT
    if is_longdouble_binary_compatible():
        FFTWDATA_LONGDOUBLE = np.load(join(fftpack_test_dir, 'fftw_longdouble_ref.npz'))
    else:
        FFTWDATA_LONGDOUBLE = {k: v.astype(np.longfloat) for k, v in FFTWDATA_DOUBLE.items()}
    ref = {'FFTWDATA_LONGDOUBLE': FFTWDATA_LONGDOUBLE, 'FFTWDATA_DOUBLE': FFTWDATA_DOUBLE, 'FFTWDATA_SINGLE': FFTWDATA_SINGLE, 'FFTWDATA_SIZES': FFTWDATA_SIZES, 'X': X, 'Y': Y}
    globals()['__reference_data'] = ref
    return ref