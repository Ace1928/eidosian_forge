from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
def test_incorrect_output(self):
    with pytest.raises(ValueError):
        MultiComparison(np.array([1] * 10), [1, 2] * 4)
    with pytest.raises(ValueError):
        MultiComparison(np.array([1] * 10), [1, 2] * 6)
    with pytest.raises(ValueError):
        MultiComparison(np.array([1] * 10), [1] * 10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert_raises(ValueError, MultiComparison, np.array([1] * 10), [1, 2] * 5, group_order=[1])
    data = np.arange(15)
    groups = np.repeat([1, 2, 3], 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mod1 = MultiComparison(np.array(data), groups, group_order=[1, 2])
        assert_equal(len(w), 1)
        assert issubclass(w[0].category, UserWarning)
    res1 = mod1.tukeyhsd(alpha=0.01)
    mod2 = MultiComparison(np.array(data[:10]), groups[:10])
    res2 = mod2.tukeyhsd(alpha=0.01)
    attributes = ['confint', 'data', 'df_total', 'groups', 'groupsunique', 'meandiffs', 'q_crit', 'reject', 'reject2', 'std_pairs', 'variance']
    for att in attributes:
        err_msg = att + 'failed'
        assert_allclose(getattr(res1, att), getattr(res2, att), rtol=1e-14, err_msg=err_msg)
    attributes = ['data', 'datali', 'groupintlab', 'groups', 'groupsunique', 'ngroups', 'nobs', 'pairindices']
    for att in attributes:
        err_msg = att + 'failed'
        assert_allclose(getattr(mod1, att), getattr(mod2, att), rtol=1e-14, err_msg=err_msg)