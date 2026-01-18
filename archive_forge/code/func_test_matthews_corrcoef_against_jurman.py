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
def test_matthews_corrcoef_against_jurman():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=20)
    y_pred = rng.randint(0, 2, size=20)
    sample_weight = rng.rand(20)
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    N = len(C)
    cov_ytyp = sum([C[k, k] * C[m, l] - C[l, k] * C[k, m] for k in range(N) for m in range(N) for l in range(N)])
    cov_ytyt = sum([C[:, k].sum() * np.sum([C[g, f] for f in range(N) for g in range(N) if f != k]) for k in range(N)])
    cov_ypyp = np.sum([C[k, :].sum() * np.sum([C[f, g] for f in range(N) for g in range(N) if f != k]) for k in range(N)])
    mcc_jurman = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
    mcc_ours = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
    assert_almost_equal(mcc_ours, mcc_jurman, 10)