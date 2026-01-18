import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_sparse_code_signal_deprecation_warning():
    """Check the message for future deprecation."""
    warn_msg = 'data_transposed was deprecated in version 1.3'
    with pytest.warns(FutureWarning, match=warn_msg):
        make_sparse_coded_signal(n_samples=1, n_components=1, n_features=1, n_nonzero_coefs=1, random_state=0, data_transposed=True)