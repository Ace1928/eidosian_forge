from statsmodels.compat.python import lzip, lmap
from numpy.testing import (
import numpy as np
import pytest
from statsmodels.stats.libqsturng import qsturng, psturng
@pytest.mark.skip
def test_all_to_tbl(self):
    from statsmodels.stats.libqsturng.make_tbls import T, R
    ps, rs, vs, qs = ([], [], [], [])
    for p in T:
        for v in T[p]:
            for r in R.keys():
                ps.append(p)
                vs.append(v)
                rs.append(r)
                qs.append(T[p][v][R[r]])
    qs = np.array(qs)
    errors = np.abs(qs - qsturng(ps, rs, vs)) / qs
    assert_equal(np.array([]), np.where(errors > 0.03)[0])