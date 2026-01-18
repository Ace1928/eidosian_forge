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
def test_evalfactor_reraise():

    def raise_patsy_error(x):
        raise PatsyError('WHEEEEEE')
    formula = 'raise_patsy_error(X) + Y'
    try:
        dmatrix(formula, {'X': [1, 2, 3], 'Y': [4, 5, 6]})
    except PatsyError as e:
        assert e.origin == Origin(formula, 0, formula.index(' '))
    else:
        assert False
    try:
        dmatrix('1 + x[1]', {'x': {}})
    except Exception as e:
        if sys.version_info[0] >= 3:
            assert isinstance(e, PatsyError)
            assert e.origin == Origin('1 + x[1]', 4, 8)
        else:
            assert isinstance(e, KeyError)
    else:
        assert False