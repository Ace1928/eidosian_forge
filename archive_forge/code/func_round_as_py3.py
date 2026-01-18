from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def round_as_py3(value):
    """
    Implement round behaviour consistent with Python 3 for use in Python 2.
    (Such that halves are rounded toward the even side.
    Only for positive numbers.)
    Adapted from https://github.com/python/cpython/blob/6b678aea1ca5c3c3728cd5a7a6eb112b2dad8553/Python/pytime.c#L71

    """
    if math.fmod(value, 1) == 0.5:
        return 2 * round(value / 2)
    else:
        return round(value)