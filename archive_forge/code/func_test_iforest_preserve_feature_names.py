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
def test_iforest_preserve_feature_names():
    """Check that feature names are preserved when contamination is not "auto".

    Feature names are required for consistency checks during scoring.

    Non-regression test for Issue #25844
    """
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    X = pd.DataFrame(data=rng.randn(4), columns=['a'])
    model = IsolationForest(random_state=0, contamination=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        model.fit(X)