import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
def test_comparemeans_convenient_interface_1d(self):
    x1_2d, x2_2d = (self.x1, self.x2)
    d1 = DescrStatsW(x1_2d)
    d2 = DescrStatsW(x2_2d)
    cm1 = CompareMeans(d1, d2)
    from statsmodels.iolib.table import SimpleTable
    for use_t in [True, False]:
        for usevar in ['pooled', 'unequal']:
            smry = cm1.summary(use_t=use_t, usevar=usevar)
            assert_(isinstance(smry, SimpleTable))
    cm2 = CompareMeans.from_data(x1_2d, x2_2d)
    assert_(str(cm1.summary()) == str(cm2.summary()))