import sys
import __future__
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.categorical import C
from patsy.contrasts import Helmert
from patsy.user_util import balanced, LookupFactor
from patsy.build import (design_matrix_builders,
from patsy.highlevel import *
from patsy.util import (have_pandas,
from patsy.origin import Origin
def test_multicolumn():
    data = {'a': ['a1', 'a2'], 'X': [[1, 2], [3, 4]], 'Y': [[1, 3], [2, 4]]}
    t('X*Y', data, 0, True, [[1, 1, 2, 1, 3, 1 * 1, 2 * 1, 1 * 3, 2 * 3], [1, 3, 4, 2, 4, 3 * 2, 4 * 2, 3 * 4, 4 * 4]], ['Intercept', 'X[0]', 'X[1]', 'Y[0]', 'Y[1]', 'X[0]:Y[0]', 'X[1]:Y[0]', 'X[0]:Y[1]', 'X[1]:Y[1]'])
    t('a:X + Y', data, 0, True, [[1, 1, 0, 2, 0, 1, 3], [1, 0, 3, 0, 4, 2, 4]], ['Intercept', 'a[a1]:X[0]', 'a[a2]:X[0]', 'a[a1]:X[1]', 'a[a2]:X[1]', 'Y[0]', 'Y[1]'])