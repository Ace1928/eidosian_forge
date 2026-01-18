from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize('metric_name', chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
def test_format_invariance(metric_name):
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7]

    def generate_formats(y):
        y = np.array(y)
        yield (y, 'array of ints')
        yield (y.tolist(), 'list of ints')
        yield ([str(x) + '-a' for x in y.tolist()], 'list of strs')
        yield (np.array([str(x) + '-a' for x in y.tolist()], dtype=object), 'array of strs')
        yield (y - 1, 'including negative ints')
        yield (y + 1, 'strictly positive ints')
    if metric_name in SUPERVISED_METRICS:
        metric = SUPERVISED_METRICS[metric_name]
        score_1 = metric(y_true, y_pred)
        y_true_gen = generate_formats(y_true)
        y_pred_gen = generate_formats(y_pred)
        for (y_true_fmt, fmt_name), (y_pred_fmt, _) in zip(y_true_gen, y_pred_gen):
            assert score_1 == metric(y_true_fmt, y_pred_fmt)
    else:
        metric = UNSUPERVISED_METRICS[metric_name]
        X = np.random.randint(10, size=(8, 10))
        score_1 = metric(X, y_true)
        assert score_1 == metric(X.astype(float), y_true)
        y_true_gen = generate_formats(y_true)
        for y_true_fmt, fmt_name in y_true_gen:
            assert score_1 == metric(X, y_true_fmt)