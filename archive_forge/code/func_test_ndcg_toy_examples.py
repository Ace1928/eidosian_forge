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
@pytest.mark.parametrize('ignore_ties', [True, False])
def test_ndcg_toy_examples(ignore_ties):
    y_true = 3 * np.eye(7)[:5]
    y_score = np.tile(np.arange(6, -1, -1), (5, 1))
    y_score_noisy = y_score + np.random.RandomState(0).uniform(-0.2, 0.2, size=y_score.shape)
    assert _dcg_sample_scores(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(y_true, y_score_noisy, ignore_ties=ignore_ties) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    assert _ndcg_sample_scores(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(1 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(y_true, y_score, log_base=10, ignore_ties=ignore_ties) == pytest.approx(3 / np.log10(np.arange(2, 7)))
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx((1 / np.log2(np.arange(2, 7))).mean())
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx((3 / np.log2(np.arange(2, 7))).mean())
    y_true = 3 * np.ones((5, 7))
    expected_dcg_score = (3 / np.log2(np.arange(2, 9))).sum()
    assert _dcg_sample_scores(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(expected_dcg_score * np.ones(5))
    assert _ndcg_sample_scores(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(np.ones(5))
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(expected_dcg_score)
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(1.0)