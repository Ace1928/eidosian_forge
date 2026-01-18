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
@pytest.mark.parametrize('Classifier, model', [(svm.SVC, {'when-left': [0.3998, 0.4], 'when-right': [0.4, 0.3999]}), (svm.NuSVC, {'when-left': [0.3333, 0.3333], 'when-right': [0.3333, 0.3333]})], ids=['SVC', 'NuSVC'])
@pytest.mark.parametrize('sample_weight, mask_side', [([1, -0.5, 1, 1, 1, 1], 'when-left'), ([1, 1, 1, 0, 1, 1], 'when-right')], ids=['partial-mask-label-1', 'partial-mask-label-2'])
def test_negative_weights_svc_leave_two_labels(Classifier, model, sample_weight, mask_side):
    clf = Classifier(kernel='linear')
    clf.fit(X, Y, sample_weight=sample_weight)
    assert_allclose(clf.coef_, [model[mask_side]], rtol=0.001)