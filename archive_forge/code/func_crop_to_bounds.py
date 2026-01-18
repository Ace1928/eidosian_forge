import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
def crop_to_bounds(self, val):
    """
        Return the given value cropped to be within the hard bounds
        for this parameter.

        If a numeric value is passed in, check it is within the hard
        bounds. If it is larger than the high bound, return the high
        bound. If it's smaller, return the low bound. In either case, the
        returned value could be None.  If a non-numeric value is passed
        in, set to be the default value (which could be None).  In no
        case is an exception raised; all values are accepted.

        As documented in https://github.com/holoviz/param/issues/80,
        currently does not respect exclusive bounds, which would
        strictly require setting to one less for integer values or
        an epsilon less for floats.
        """
    if _is_number(val):
        if self.bounds is None:
            return val
        vmin, vmax = self.bounds
        if vmin is not None:
            if val < vmin:
                return vmin
        if vmax is not None:
            if val > vmax:
                return vmax
    elif self.allow_None and val is None:
        return val
    else:
        return self.default
    return val