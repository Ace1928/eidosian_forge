import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('method', ['fit', 'fit_transform', 'transform'])
def test_additive_chi2_sampler_wrong_sample_steps(method):
    """Check that we raise a ValueError on invalid sample_steps"""
    transformer = AdditiveChi2Sampler(sample_steps=4)
    msg = re.escape('If sample_steps is not in [1, 2, 3], you need to provide sample_interval')
    with pytest.raises(ValueError, match=msg):
        getattr(transformer, method)(X)