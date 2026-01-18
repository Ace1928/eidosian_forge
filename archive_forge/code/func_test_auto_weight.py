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
@ignore_warnings(category=UndefinedMetricWarning)
def test_auto_weight():
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils import compute_class_weight
    X, y = (iris.data[:, :2], iris.target + 1)
    unbalanced = np.delete(np.arange(y.size), np.where(y > 2)[0][::2])
    classes = np.unique(y[unbalanced])
    class_weights = compute_class_weight('balanced', classes=classes, y=y[unbalanced])
    assert np.argmax(class_weights) == 2
    for clf in (svm.SVC(kernel='linear'), svm.LinearSVC(dual='auto', random_state=0), LogisticRegression()):
        y_pred = clf.fit(X[unbalanced], y[unbalanced]).predict(X)
        clf.set_params(class_weight='balanced')
        y_pred_balanced = clf.fit(X[unbalanced], y[unbalanced]).predict(X)
        assert metrics.f1_score(y, y_pred, average='macro') <= metrics.f1_score(y, y_pred_balanced, average='macro')