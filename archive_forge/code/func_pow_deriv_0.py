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
def pow_deriv_0(x, y):
    if y == 0:
        return 0.0
    elif x != 0 or y % 1 == 0:
        return y * x ** (y - 1)
    else:
        return float('nan')