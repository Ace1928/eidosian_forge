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
def lt_on_aff_funcs(self, y_with_uncert):
    """
    __lt__ operator, assuming that both self and y_with_uncert are
    AffineScalarFunc objects.
    """
    return self._nominal_value < y_with_uncert._nominal_value