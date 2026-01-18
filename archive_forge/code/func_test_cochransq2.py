from statsmodels.compat.python import lzip
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
from scipy import stats
import pytest
from statsmodels.stats.contingency_tables import (
from statsmodels.sandbox.stats.runs import (Runs,
from statsmodels.sandbox.stats.runs import mcnemar as sbmcnemar
from statsmodels.stats.nonparametric import (
from statsmodels.tools.testing import Holder
def test_cochransq2():
    data = np.array('\n        0 0 0 1\n        0 0 0 1\n        0 0 0 1\n        1 1 1 1\n        1 0 0 1\n        0 1 0 1\n        1 0 0 1\n        0 0 0 1\n        0 1 0 0\n        0 0 0 0\n        1 0 0 1\n        0 0 1 1'.split(), int).reshape(-1, 4)
    res = cochrans_q(data)
    assert_allclose([res.statistic, res.pvalue], [13.2857143, 0.00405776], rtol=1e-06)