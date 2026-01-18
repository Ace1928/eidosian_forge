import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
def test_micro_averaged_ovr_roc_auc(global_random_seed):
    seed = global_random_seed
    y_pred = stats.dirichlet.rvs([2.0, 1.0, 0.5], size=1000, random_state=seed)
    y_true = np.asarray([stats.multinomial.rvs(n=1, p=y_pred_i, random_state=seed).argmax() for y_pred_i in y_pred])
    y_onehot = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, _ = roc_curve(y_onehot.ravel(), y_pred.ravel())
    roc_auc_by_hand = auc(fpr, tpr)
    roc_auc_auto = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
    assert roc_auc_by_hand == pytest.approx(roc_auc_auto)