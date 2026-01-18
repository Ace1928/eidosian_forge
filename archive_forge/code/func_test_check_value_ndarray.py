import copy
import copyreg
import io
import pickle
import struct
from itertools import chain, product
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose
from sklearn import clone, datasets, tree
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import _sparse_random_matrix
from sklearn.tree import (
from sklearn.tree._classes import (
from sklearn.tree._tree import (
from sklearn.tree._tree import Tree as CythonTree
from sklearn.utils import _IS_32BIT, compute_sample_weight
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import check_sample_weights_invariance
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
def test_check_value_ndarray():
    expected_dtype = np.dtype(np.float64)
    expected_shape = (5, 1, 2)
    value_ndarray = np.zeros(expected_shape, dtype=expected_dtype)
    allowed_dtypes = [expected_dtype, expected_dtype.newbyteorder()]
    for dt in allowed_dtypes:
        _check_value_ndarray(value_ndarray, expected_dtype=dt, expected_shape=expected_shape)
    with pytest.raises(ValueError, match='Wrong shape.+value array'):
        _check_value_ndarray(value_ndarray, expected_dtype=expected_dtype, expected_shape=(1, 2))
    for problematic_arr in [value_ndarray[:, :, :1], np.asfortranarray(value_ndarray)]:
        with pytest.raises(ValueError, match='value array.+C-contiguous'):
            _check_value_ndarray(problematic_arr, expected_dtype=expected_dtype, expected_shape=problematic_arr.shape)
    with pytest.raises(ValueError, match='value array.+incompatible dtype'):
        _check_value_ndarray(value_ndarray.astype(np.float32), expected_dtype=expected_dtype, expected_shape=expected_shape)