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
@pytest.mark.parametrize('y_true,y_pred', [(['a', 'b', 'a', 'b'], ['a', 'a', 'a', 'b']), (['a', 'b', 'c', 'b'], ['a', 'a', 'a', 'b']), (['a', 'a', 'a', 'b'], ['a', 'b', 'c', 'b'])])
def test_balanced_accuracy_score(y_true, y_pred):
    macro_recall = recall_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
    with ignore_warnings():
        balanced = balanced_accuracy_score(y_true, y_pred)
    assert balanced == pytest.approx(macro_recall)
    adjusted = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[0]))
    assert adjusted == (balanced - chance) / (1 - chance)