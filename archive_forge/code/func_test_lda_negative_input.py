import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.linalg import block_diag
from scipy.special import psi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._online_lda_fast import (
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_lda_negative_input():
    X = np.full((5, 10), -1.0)
    lda = LatentDirichletAllocation()
    regex = '^Negative values in data passed'
    with pytest.raises(ValueError, match=regex):
        lda.fit(X)