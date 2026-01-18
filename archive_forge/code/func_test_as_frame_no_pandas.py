from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import assert_allclose_dense_sparse
def test_as_frame_no_pandas(fetch_20newsgroups_vectorized_fxt, hide_available_pandas):
    check_pandas_dependency_message(fetch_20newsgroups_vectorized_fxt)