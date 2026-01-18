import math
import numbers
import random
import sys
from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from functools import reduce
from operator import mul
from collections import Counter, namedtuple, defaultdict
def overlap(self, other):
    """Compute the overlapping coefficient (OVL) between two normal distributions.

        Measures the agreement between two normal probability distributions.
        Returns a value between 0.0 and 1.0 giving the overlapping area in
        the two underlying probability density functions.

            >>> N1 = NormalDist(2.4, 1.6)
            >>> N2 = NormalDist(3.2, 2.0)
            >>> N1.overlap(N2)
            0.8035050657330205
        """
    if not isinstance(other, NormalDist):
        raise TypeError('Expected another NormalDist instance')
    X, Y = (self, other)
    if (Y._sigma, Y._mu) < (X._sigma, X._mu):
        X, Y = (Y, X)
    X_var, Y_var = (X.variance, Y.variance)
    if not X_var or not Y_var:
        raise StatisticsError('overlap() not defined when sigma is zero')
    dv = Y_var - X_var
    dm = fabs(Y._mu - X._mu)
    if not dv:
        return 1.0 - erf(dm / (2.0 * X._sigma * _SQRT2))
    a = X._mu * Y_var - Y._mu * X_var
    b = X._sigma * Y._sigma * sqrt(dm * dm + dv * log(Y_var / X_var))
    x1 = (a + b) / dv
    x2 = (a - b) / dv
    return 1.0 - (fabs(Y.cdf(x1) - X.cdf(x1)) + fabs(Y.cdf(x2) - X.cdf(x2)))