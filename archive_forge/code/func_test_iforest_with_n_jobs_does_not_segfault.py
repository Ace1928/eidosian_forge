import warnings
from unittest.mock import Mock, patch
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_iforest_with_n_jobs_does_not_segfault(csc_container):
    """Check that Isolation Forest does not segfault with n_jobs=2

    Non-regression test for #23252
    """
    X, _ = make_classification(n_samples=85000, n_features=100, random_state=0)
    X = csc_container(X)
    IsolationForest(n_estimators=10, max_samples=256, n_jobs=2).fit(X)