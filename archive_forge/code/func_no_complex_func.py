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
def no_complex_func(*args, **kwargs):
    '\n            Like %s, but raises a ValueError exception if the result\n            is complex.\n            ' % func.__name__
    value = func(*args, **kwargs)
    if isinstance(value, complex):
        raise ValueError('The uncertainties module does not handle complex results')
    else:
        return value