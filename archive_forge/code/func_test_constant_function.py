import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
def test_constant_function(self, ishigami_ref_indices):

    def f_ishigami_vec_const(x):
        """Output of shape (3, n)."""
        res = f_ishigami(x)
        return (res, res * 0 + 10, res)
    rng = np.random.default_rng(28631265345463262246170309650372465332)
    res = sobol_indices(func=f_ishigami_vec_const, n=4096, dists=self.dists, random_state=rng)
    ishigami_vec_indices = [[ishigami_ref_indices[0], [0, 0, 0], ishigami_ref_indices[0]], [ishigami_ref_indices[1], [0, 0, 0], ishigami_ref_indices[1]]]
    assert_allclose(res.first_order, ishigami_vec_indices[0], atol=0.01)
    assert_allclose(res.total_order, ishigami_vec_indices[1], atol=0.01)