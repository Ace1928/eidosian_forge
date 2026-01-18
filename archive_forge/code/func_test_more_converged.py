import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
@pytest.mark.xfail_on_32bit("Can't create large array for test")
def test_more_converged(self, ishigami_ref_indices):
    rng = np.random.default_rng(28631265345463262246170309650372465332)
    res = sobol_indices(func=f_ishigami, n=2 ** 19, dists=self.dists, random_state=rng)
    assert_allclose(res.first_order, ishigami_ref_indices[0], atol=0.0001)
    assert_allclose(res.total_order, ishigami_ref_indices[1], atol=0.0001)