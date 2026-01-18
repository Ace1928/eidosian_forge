import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('is_categorical, known_categories, match', [(np.array([True]), [None], 'Known categories for feature 0 must be provided'), (np.array([False]), np.array([1, 2, 3]), "isn't marked as a categorical feature, but categories were passed")])
def test_categorical_parameters(is_categorical, known_categories, match):
    X = np.array([[1, 2, 3]], dtype=X_DTYPE)
    bin_mapper = _BinMapper(is_categorical=is_categorical, known_categories=known_categories)
    with pytest.raises(ValueError, match=match):
        bin_mapper.fit(X)