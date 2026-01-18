from __future__ import print_function
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import (atleast_2d_column_default,
from patsy.desc import Term, INTERCEPT
from patsy.build import *
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo
def all_subsets(l):
    if not l:
        yield tuple()
    else:
        obj = l[0]
        for subset in all_subsets(l[1:]):
            yield tuple(sorted(subset))
            yield tuple(sorted((obj,) + subset))