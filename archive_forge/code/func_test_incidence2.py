import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_incidence2():
    np.random.seed(2423)
    n = 200
    time = -np.log(np.random.uniform(size=n))
    status = np.random.randint(0, 3, size=n)
    ii = np.argsort(time)
    time = time[ii]
    status = status[ii]
    ci = CumIncidenceRight(time, status)
    statusa = 1 * (status >= 1)
    sf = SurvfuncRight(time, statusa)
    x = 1 - sf.surv_prob
    y = (ci.cinc[0] + ci.cinc[1])[np.flatnonzero(statusa)]
    assert_allclose(x, y)