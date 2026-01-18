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
def test_dmatrix_dmatrices_no_data():
    x = [1, 2, 3]
    y = [4, 5, 6]
    assert np.allclose(dmatrix('x'), [[1, 1], [1, 2], [1, 3]])
    lhs, rhs = dmatrices('y ~ x')
    assert np.allclose(lhs, [[4], [5], [6]])
    assert np.allclose(rhs, [[1, 1], [1, 2], [1, 3]])