import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_incidence():
    ftime = np.r_[1, 1, 2, 4, 4, 4, 6, 6, 7, 8, 9, 9, 9, 1, 2, 2, 4, 4]
    fstat = np.r_[1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ci = CumIncidenceRight(ftime, fstat)
    cinc = [np.array([0.11111111, 0.17037037, 0.17037037, 0.17037037, 0.17037037, 0.17037037, 0.17037037]), np.array([0.0, 0.0, 0.20740741, 0.20740741, 0.20740741, 0.20740741, 0.20740741]), np.array([0.0, 0.0, 0.0, 0.17777778, 0.26666667, 0.26666667, 0.26666667])]
    assert_allclose(cinc[0], ci.cinc[0])
    assert_allclose(cinc[1], ci.cinc[1])
    assert_allclose(cinc[2], ci.cinc[2])
    cinc_se = [np.array([0.07407407, 0.08976251, 0.08976251, 0.08976251, 0.08976251, 0.08976251, 0.08976251]), np.array([0.0, 0.0, 0.10610391, 0.10610391, 0.10610391, 0.10610391, 0.10610391]), np.array([0.0, 0.0, 0.0, 0.11196147, 0.12787781, 0.12787781, 0.12787781])]
    assert_allclose(cinc_se[0], ci.cinc_se[0])
    assert_allclose(cinc_se[1], ci.cinc_se[1])
    assert_allclose(cinc_se[2], ci.cinc_se[2])
    weights = np.ones(len(ftime))
    ciw = CumIncidenceRight(ftime, fstat, freq_weights=weights)
    assert_allclose(ci.cinc[0], ciw.cinc[0])
    assert_allclose(ci.cinc[1], ciw.cinc[1])
    assert_allclose(ci.cinc[2], ciw.cinc[2])