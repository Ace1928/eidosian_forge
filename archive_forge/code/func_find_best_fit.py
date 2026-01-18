import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def find_best_fit(self, r, nr):
    """
        Use simple linear regression to tune parameters self._slope and
        self._intercept in the log-log space based on count and Nr(count)
        (Work in log space to avoid floating point underflow.)
        """
    if not r or not nr:
        return
    zr = []
    for j in range(len(r)):
        i = r[j - 1] if j > 0 else 0
        k = 2 * r[j] - i if j == len(r) - 1 else r[j + 1]
        zr_ = 2.0 * nr[j] / (k - i)
        zr.append(zr_)
    log_r = [math.log(i) for i in r]
    log_zr = [math.log(i) for i in zr]
    xy_cov = x_var = 0.0
    x_mean = sum(log_r) / len(log_r)
    y_mean = sum(log_zr) / len(log_zr)
    for x, y in zip(log_r, log_zr):
        xy_cov += (x - x_mean) * (y - y_mean)
        x_var += (x - x_mean) ** 2
    self._slope = xy_cov / x_var if x_var != 0 else 0.0
    if self._slope >= -1:
        warnings.warn('SimpleGoodTuring did not find a proper best fit line for smoothing probabilities of occurrences. The probability estimates are likely to be unreliable.')
    self._intercept = y_mean - self._slope * x_mean