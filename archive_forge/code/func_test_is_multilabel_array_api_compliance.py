from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
@pytest.mark.parametrize('array_namespace, device, dtype_name', yield_namespace_device_dtype_combinations())
def test_is_multilabel_array_api_compliance(array_namespace, device, dtype_name):
    xp = _array_api_for_tests(array_namespace, device)
    for group, group_examples in ARRAY_API_EXAMPLES.items():
        dense_exp = group == 'multilabel-indicator'
        for example in group_examples:
            if np.asarray(example).dtype.kind == 'f':
                example = np.asarray(example, dtype=dtype_name)
            else:
                example = np.asarray(example)
            example = xp.asarray(example, device=device)
            with config_context(array_api_dispatch=True):
                assert dense_exp == is_multilabel(example), f'is_multilabel({example!r}) should be {dense_exp}'