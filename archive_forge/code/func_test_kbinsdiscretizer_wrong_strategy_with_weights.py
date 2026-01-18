import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('strategy', ['uniform'])
def test_kbinsdiscretizer_wrong_strategy_with_weights(strategy):
    """Check that we raise an error when the wrong strategy is used."""
    sample_weight = np.ones(shape=len(X))
    est = KBinsDiscretizer(n_bins=3, strategy=strategy)
    err_msg = "`sample_weight` was provided but it cannot be used with strategy='uniform'."
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, sample_weight=sample_weight)