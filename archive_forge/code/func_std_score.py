from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def std_score(self, value):
    """
        Return 'value' - nominal value, in units of the standard
        deviation.

        Raises a ValueError exception if the standard deviation is zero.
        """
    try:
        return (value - self._nominal_value) / self.std_dev
    except ZeroDivisionError:
        raise ValueError('The standard deviation is zero: undefined result')