from warnings import warn as _warn
from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
from math import sqrt as _sqrt, acos as _acos, cos as _cos, sin as _sin
from math import tau as TWOPI, floor as _floor, isfinite as _isfinite
from os import urandom as _urandom
from _collections_abc import Set as _Set, Sequence as _Sequence
from operator import index as _index
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import os as _os
import _random
def gammavariate(self, alpha, beta):
    """Gamma distribution.  Not the gamma function!

        Conditions on the parameters are alpha > 0 and beta > 0.

        The probability distribution function is:

                    x ** (alpha - 1) * math.exp(-x / beta)
          pdf(x) =  --------------------------------------
                      math.gamma(alpha) * beta ** alpha

        """
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError('gammavariate: alpha and beta must be > 0.0')
    random = self.random
    if alpha > 1.0:
        ainv = _sqrt(2.0 * alpha - 1.0)
        bbb = alpha - LOG4
        ccc = alpha + ainv
        while True:
            u1 = random()
            if not 1e-07 < u1 < 0.9999999:
                continue
            u2 = 1.0 - random()
            v = _log(u1 / (1.0 - u1)) / ainv
            x = alpha * _exp(v)
            z = u1 * u1 * u2
            r = bbb + ccc * v - x
            if r + SG_MAGICCONST - 4.5 * z >= 0.0 or r >= _log(z):
                return x * beta
    elif alpha == 1.0:
        return -_log(1.0 - random()) * beta
    else:
        while True:
            u = random()
            b = (_e + alpha) / _e
            p = b * u
            if p <= 1.0:
                x = p ** (1.0 / alpha)
            else:
                x = -_log((b - p) / alpha)
            u1 = random()
            if p > 1.0:
                if u1 <= x ** (alpha - 1.0):
                    break
            elif u1 <= _exp(-x):
                break
        return x * beta