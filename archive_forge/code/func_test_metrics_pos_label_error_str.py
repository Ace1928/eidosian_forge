from functools import partial
from inspect import signature
from itertools import chain, permutations, product
import numpy as np
import pytest
from sklearn._config import config_context
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import (
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_random_state
@pytest.mark.parametrize('metric, y_pred_threshold', [(average_precision_score, True), (brier_score_loss, True), (f1_score, False), (partial(fbeta_score, beta=1), False), (jaccard_score, False), (precision_recall_curve, True), (precision_score, False), (recall_score, False), (roc_curve, True)])
@pytest.mark.parametrize('dtype_y_str', [str, object])
def test_metrics_pos_label_error_str(metric, y_pred_threshold, dtype_y_str):
    rng = np.random.RandomState(42)
    y1 = np.array(['spam'] * 3 + ['eggs'] * 2, dtype=dtype_y_str)
    y2 = rng.randint(0, 2, size=y1.size)
    if not y_pred_threshold:
        y2 = np.array(['spam', 'eggs'], dtype=dtype_y_str)[y2]
    err_msg_pos_label_None = "y_true takes value in {'eggs', 'spam'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicit"
    err_msg_pos_label_1 = "pos_label=1 is not a valid label. It should be one of \\['eggs', 'spam'\\]"
    pos_label_default = signature(metric).parameters['pos_label'].default
    err_msg = err_msg_pos_label_1 if pos_label_default == 1 else err_msg_pos_label_None
    with pytest.raises(ValueError, match=err_msg):
        metric(y1, y2)