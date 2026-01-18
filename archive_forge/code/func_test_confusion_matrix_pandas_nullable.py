import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean'])
def test_confusion_matrix_pandas_nullable(dtype):
    """Checks that confusion_matrix works with pandas nullable dtypes.

    Non-regression test for gh-25635.
    """
    pd = pytest.importorskip('pandas')
    y_ndarray = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1])
    y_true = pd.Series(y_ndarray, dtype=dtype)
    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype='int64')
    output = confusion_matrix(y_true, y_predicted)
    expected_output = confusion_matrix(y_ndarray, y_predicted)
    assert_array_equal(output, expected_output)