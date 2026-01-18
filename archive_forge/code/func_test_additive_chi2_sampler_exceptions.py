import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_additive_chi2_sampler_exceptions():
    """Ensures correct error message"""
    transformer = AdditiveChi2Sampler()
    X_neg = X.copy()
    X_neg[0, 0] = -1
    with pytest.raises(ValueError, match='X in AdditiveChi2Sampler.fit'):
        transformer.fit(X_neg)
    with pytest.raises(ValueError, match='X in AdditiveChi2Sampler.transform'):
        transformer.fit(X)
        transformer.transform(X_neg)