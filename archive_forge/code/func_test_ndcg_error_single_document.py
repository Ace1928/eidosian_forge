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
def test_ndcg_error_single_document():
    """Check that we raise an informative error message when trying to
    compute NDCG with a single document."""
    err_msg = 'Computing NDCG is only meaningful when there is more than 1 document. Got 1 instead.'
    with pytest.raises(ValueError, match=err_msg):
        ndcg_score([[1]], [[1]])