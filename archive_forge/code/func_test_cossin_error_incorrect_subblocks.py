import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs
def test_cossin_error_incorrect_subblocks():
    with pytest.raises(ValueError, match='be due to missing p, q arguments.'):
        cossin(([1, 2], [3, 4, 5], [6, 7], [8, 9, 10]))