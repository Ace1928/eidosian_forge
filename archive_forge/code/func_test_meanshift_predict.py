import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_meanshift_predict(global_dtype):
    ms = MeanShift(bandwidth=1.2)
    X_with_global_dtype = X.astype(global_dtype, copy=False)
    labels = ms.fit_predict(X_with_global_dtype)
    labels2 = ms.predict(X_with_global_dtype)
    assert_array_equal(labels, labels2)