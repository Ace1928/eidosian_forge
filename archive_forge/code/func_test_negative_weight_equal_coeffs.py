import re
import numpy as np
import pytest
from numpy.testing import (
from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (  # type: ignore
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('Estimator', [svm.SVC, svm.NuSVC, svm.NuSVR], ids=['SVC', 'NuSVC', 'NuSVR'])
@pytest.mark.parametrize('sample_weight', [[1, -0.5, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1]], ids=['partial-mask-label-1', 'partial-mask-label-2'])
def test_negative_weight_equal_coeffs(Estimator, sample_weight):
    est = Estimator(kernel='linear')
    est.fit(X, Y, sample_weight=sample_weight)
    coef = np.abs(est.coef_).ravel()
    assert coef[0] == pytest.approx(coef[1], rel=0.001)