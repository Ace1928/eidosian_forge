import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
def test_cv_pipeline_precomputed():
    X = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1]])
    y_true = np.ones((4,))
    K = X.dot(X.T)
    kcent = KernelCenterer()
    pipeline = Pipeline([('kernel_centerer', kcent), ('svr', SVR())])
    assert pipeline._get_tags()['pairwise']
    y_pred = cross_val_predict(pipeline, K, y_true, cv=2)
    assert_array_almost_equal(y_true, y_pred)