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
def robust_align(orig_str, fill_char, align_option, width):
    """
    Aligns the given string with the given fill character.

    orig_str -- string to be aligned (str or unicode object).

    fill_char -- if empty, space is used.

    align_option -- as accepted by format().

    wdith -- string that contains the width.
    """
    return format(orig_str, fill_char + align_option + width)