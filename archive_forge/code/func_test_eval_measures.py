import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.tools.eval_measures import (
def test_eval_measures():
    x = np.arange(20).reshape(4, 5)
    y = np.ones((4, 5))
    assert_equal(iqr(x, y), 5 * np.ones(5))
    assert_equal(iqr(x, y, axis=1), 2 * np.ones(4))
    assert_equal(iqr(x, y, axis=None), 9)
    assert_equal(mse(x, y), np.array([73.5, 87.5, 103.5, 121.5, 141.5]))
    assert_equal(mse(x, y, axis=1), np.array([3.0, 38.0, 123.0, 258.0]))
    assert_almost_equal(rmse(x, y), np.array([8.5732141, 9.35414347, 10.17349497, 11.02270384, 11.89537725]))
    assert_almost_equal(rmse(x, y, axis=1), np.array([1.73205081, 6.164414, 11.09053651, 16.0623784]))
    err = x - y
    loc = np.where(x != 0)
    err[loc] /= x[loc]
    err[np.where(x == 0)] = np.nan
    expected = np.sqrt(np.nanmean(err ** 2, 0) * 100)
    assert_almost_equal(rmspe(x, y), expected)
    err[np.where(np.isnan(err))] = 0.0
    expected = np.sqrt(np.nanmean(err ** 2, 0) * 100)
    assert_almost_equal(rmspe(x, y, zeros=0), expected)
    assert_equal(maxabs(x, y), np.array([14.0, 15.0, 16.0, 17.0, 18.0]))
    assert_equal(maxabs(x, y, axis=1), np.array([3.0, 8.0, 13.0, 18.0]))
    assert_equal(meanabs(x, y), np.array([7.0, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(meanabs(x, y, axis=1), np.array([1.4, 6.0, 11.0, 16.0]))
    assert_equal(meanabs(x, y, axis=0), np.array([7.0, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(medianabs(x, y), np.array([6.5, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(medianabs(x, y, axis=1), np.array([1.0, 6.0, 11.0, 16.0]))
    assert_equal(bias(x, y), np.array([6.5, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(bias(x, y, axis=1), np.array([1.0, 6.0, 11.0, 16.0]))
    assert_equal(medianbias(x, y), np.array([6.5, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(medianbias(x, y, axis=1), np.array([1.0, 6.0, 11.0, 16.0]))
    assert_equal(vare(x, y), np.array([31.25, 31.25, 31.25, 31.25, 31.25]))
    assert_equal(vare(x, y, axis=1), np.array([2.0, 2.0, 2.0, 2.0]))