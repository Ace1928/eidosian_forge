import timeit
from functools import reduce
import numpy as np
from numpy import float_
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import build_err_msg
@np.errstate(all='ignore')
def test_99(self):
    ott = self.array([0.0, 1.0, 2.0, 3.0], mask=[1, 0, 0, 0])
    self.assert_array_equal(2.0, self.average(ott, axis=0))
    self.assert_array_equal(2.0, self.average(ott, weights=[1.0, 1.0, 2.0, 1.0]))
    result, wts = self.average(ott, weights=[1.0, 1.0, 2.0, 1.0], returned=1)
    self.assert_array_equal(2.0, result)
    assert wts == 4.0
    ott[:] = self.masked
    assert self.average(ott, axis=0) is self.masked
    ott = self.array([0.0, 1.0, 2.0, 3.0], mask=[1, 0, 0, 0])
    ott = ott.reshape(2, 2)
    ott[:, 1] = self.masked
    self.assert_array_equal(self.average(ott, axis=0), [2.0, 0.0])
    assert self.average(ott, axis=1)[0] is self.masked
    self.assert_array_equal([2.0, 0.0], self.average(ott, axis=0))
    result, wts = self.average(ott, axis=0, returned=1)
    self.assert_array_equal(wts, [1.0, 0.0])
    w1 = [0, 1, 1, 1, 1, 0]
    w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
    x = self.arange(6)
    self.assert_array_equal(self.average(x, axis=0), 2.5)
    self.assert_array_equal(self.average(x, axis=0, weights=w1), 2.5)
    y = self.array([self.arange(6), 2.0 * self.arange(6)])
    self.assert_array_equal(self.average(y, None), np.add.reduce(np.arange(6)) * 3.0 / 12.0)
    self.assert_array_equal(self.average(y, axis=0), np.arange(6) * 3.0 / 2.0)
    self.assert_array_equal(self.average(y, axis=1), [self.average(x, axis=0), self.average(x, axis=0) * 2.0])
    self.assert_array_equal(self.average(y, None, weights=w2), 20.0 / 6.0)
    self.assert_array_equal(self.average(y, axis=0, weights=w2), [0.0, 1.0, 2.0, 3.0, 4.0, 10.0])
    self.assert_array_equal(self.average(y, axis=1), [self.average(x, axis=0), self.average(x, axis=0) * 2.0])
    m1 = self.zeros(6)
    m2 = [0, 0, 1, 1, 0, 0]
    m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
    m4 = self.ones(6)
    m5 = [0, 1, 1, 1, 1, 1]
    self.assert_array_equal(self.average(self.masked_array(x, m1), axis=0), 2.5)
    self.assert_array_equal(self.average(self.masked_array(x, m2), axis=0), 2.5)
    self.assert_array_equal(self.average(self.masked_array(x, m5), axis=0), 0.0)
    self.assert_array_equal(self.count(self.average(self.masked_array(x, m4), axis=0)), 0)
    z = self.masked_array(y, m3)
    self.assert_array_equal(self.average(z, None), 20.0 / 6.0)
    self.assert_array_equal(self.average(z, axis=0), [0.0, 1.0, 99.0, 99.0, 4.0, 7.5])
    self.assert_array_equal(self.average(z, axis=1), [2.5, 5.0])
    self.assert_array_equal(self.average(z, axis=0, weights=w2), [0.0, 1.0, 99.0, 99.0, 4.0, 10.0])