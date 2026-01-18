import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs
def test_cossin_error_partitioning():
    x = np.array(ortho_group.rvs(4), dtype=np.float64)
    with pytest.raises(ValueError, match='invalid p=0.*0<p<4.*'):
        cossin(x, 0, 1)
    with pytest.raises(ValueError, match='invalid p=4.*0<p<4.*'):
        cossin(x, 4, 1)
    with pytest.raises(ValueError, match='invalid q=-2.*0<q<4.*'):
        cossin(x, 1, -2)
    with pytest.raises(ValueError, match='invalid q=5.*0<q<4.*'):
        cossin(x, 1, 5)