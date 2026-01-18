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
def signif_dgt_to_limit(value, num_signif_d):
    """
    Return the precision limit necessary to display value with
    num_signif_d significant digits.

    The precision limit is given as -1 for 1 digit after the decimal
    point, 0 for integer rounding, etc. It can be positive.
    """
    fst_digit = first_digit(value)
    limit_no_rounding = fst_digit - num_signif_d + 1
    rounded = round(value, -limit_no_rounding)
    fst_digit_rounded = first_digit(rounded)
    if fst_digit_rounded > fst_digit:
        limit_no_rounding += 1
    return limit_no_rounding