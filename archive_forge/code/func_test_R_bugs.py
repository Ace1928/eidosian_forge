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
def test_R_bugs():
    data = balanced(a=2, b=2, c=2)
    data['x'] = np.linspace(0, 1, len(data['a']))
    make_matrix(data, 4, [[], ['a', 'b']])
    make_matrix(data, 6, [['a', 'x'], ['a', 'b']])
    make_matrix(data, 6, [['a', 'c'], ['a', 'b']])