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
@pytest.mark.parametrize('loss', ['hinge', 'squared_hinge'])
@pytest.mark.parametrize('penalty', ['l1', 'l2'])
@pytest.mark.parametrize('dual', [True, False])
def test_linearsvc_parameters(loss, penalty, dual):
    X, y = make_classification(n_samples=5, n_features=5, random_state=0)
    clf = svm.LinearSVC(penalty=penalty, loss=loss, dual=dual, random_state=0)
    if (loss, penalty) == ('hinge', 'l1') or (loss, penalty, dual) == ('hinge', 'l2', False) or (penalty, dual) == ('l1', True):
        with pytest.raises(ValueError, match="Unsupported set of arguments.*penalty='%s.*loss='%s.*dual=%s" % (penalty, loss, dual)):
            clf.fit(X, y)
    else:
        clf.fit(X, y)