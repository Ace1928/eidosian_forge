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
def test_check_node_ndarray():
    expected_dtype = NODE_DTYPE
    node_ndarray = np.zeros((5,), dtype=expected_dtype)
    valid_node_ndarrays = [node_ndarray, get_different_bitness_node_ndarray(node_ndarray), get_different_alignment_node_ndarray(node_ndarray)]
    valid_node_ndarrays += [arr.astype(arr.dtype.newbyteorder()) for arr in valid_node_ndarrays]
    for arr in valid_node_ndarrays:
        _check_node_ndarray(node_ndarray, expected_dtype=expected_dtype)
    with pytest.raises(ValueError, match='Wrong dimensions.+node array'):
        problematic_node_ndarray = np.zeros((5, 2), dtype=expected_dtype)
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)
    with pytest.raises(ValueError, match='node array.+C-contiguous'):
        problematic_node_ndarray = node_ndarray[::2]
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)
    dtype_dict = {name: dtype for name, (dtype, _) in node_ndarray.dtype.fields.items()}
    new_dtype_dict = dtype_dict.copy()
    new_dtype_dict['threshold'] = np.int64
    new_dtype = np.dtype({'names': list(new_dtype_dict.keys()), 'formats': list(new_dtype_dict.values())})
    problematic_node_ndarray = node_ndarray.astype(new_dtype)
    with pytest.raises(ValueError, match='node array.+incompatible dtype'):
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)
    new_dtype_dict = dtype_dict.copy()
    new_dtype_dict['left_child'] = np.float64
    new_dtype = np.dtype({'names': list(new_dtype_dict.keys()), 'formats': list(new_dtype_dict.values())})
    problematic_node_ndarray = node_ndarray.astype(new_dtype)
    with pytest.raises(ValueError, match='node array.+incompatible dtype'):
        _check_node_ndarray(problematic_node_ndarray, expected_dtype=expected_dtype)