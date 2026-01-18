import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_adnorm():
    st_pv = []
    st_pv_R = np.array([0.5867235358882148, 0.1115380760041617])
    ad = normal_ad(x)
    assert_almost_equal(ad, st_pv_R, 12)
    st_pv.append(st_pv_R)
    st_pv_R = np.array([2.976266267594575, 8.753003709960645e-08])
    ad = normal_ad(x ** 2)
    assert_almost_equal(ad, st_pv_R, 11)
    st_pv.append(st_pv_R)
    st_pv_R = np.array([0.4892557856308528, 0.1968040759316307])
    ad = normal_ad(np.log(x ** 2))
    assert_almost_equal(ad, st_pv_R, 12)
    st_pv.append(st_pv_R)
    st_pv_R = np.array([1.459901465428267, 0.0006380009232897535])
    ad = normal_ad(np.exp(-x ** 2))
    assert_almost_equal(ad, st_pv_R, 12)
    st_pv.append(st_pv_R)
    ad = normal_ad(np.column_stack((x, x ** 2, np.log(x ** 2), np.exp(-x ** 2))).T, axis=1)
    assert_almost_equal(ad, np.column_stack(st_pv), 11)