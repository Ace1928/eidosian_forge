import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
@pytest.mark.skipif(sp_version >= parse_version('1.8'), reason='sparse arrays are available as of scipy 1.8.0')
@pytest.mark.parametrize('constructor_name', ['sparse_csr_array', 'sparse_csc_array'])
@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
def test_convert_container_raise_when_sparray_not_available(constructor_name, dtype):
    """Check that if we convert to sparse array but sparse array are not supported
    (scipy<1.8.0), we should raise an explicit error."""
    container = [0, 1]
    with pytest.raises(ValueError, match=f'only available with scipy>=1.8.0, got {sp_version}'):
        _convert_container(container, constructor_name, dtype=dtype)