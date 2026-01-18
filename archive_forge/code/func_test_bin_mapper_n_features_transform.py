import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_bin_mapper_n_features_transform():
    mapper = _BinMapper(n_bins=42, random_state=42).fit(DATA)
    err_msg = 'This estimator was fitted with 2 features but 4 got passed'
    with pytest.raises(ValueError, match=err_msg):
        mapper.transform(np.repeat(DATA, 2, axis=1))