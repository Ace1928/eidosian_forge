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
def betavariate(self, alpha, beta):
    """Beta distribution.

        Conditions on the parameters are alpha > 0 and beta > 0.
        Returned values range between 0 and 1.

        """
    y = self.gammavariate(alpha, 1.0)
    if y:
        return y / (y + self.gammavariate(beta, 1.0))
    return 0.0