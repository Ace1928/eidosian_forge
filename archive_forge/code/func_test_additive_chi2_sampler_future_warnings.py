import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_additive_chi2_sampler_future_warnings():
    """Check that we raise a FutureWarning when accessing to `sample_interval_`."""
    transformer = AdditiveChi2Sampler()
    transformer.fit(X)
    msg = re.escape('The ``sample_interval_`` attribute was deprecated in version 1.3 and will be removed 1.5.')
    with pytest.warns(FutureWarning, match=msg):
        assert transformer.sample_interval_ is not None