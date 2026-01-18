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
def test_0d_data():
    data_0d = {'x1': 1.1, 'x2': 1.2, 'a': 'a1'}
    for formula, expected in [('x1 + x2', [[1, 1.1, 1.2]]), ("C(a, levels=('a1', 'a2')) + x1", [[1, 0, 1.1]])]:
        mat = dmatrix(formula, data_0d)
        assert np.allclose(mat, expected)
        assert np.allclose(build_design_matrices([mat.design_info], data_0d)[0], expected)
        if have_pandas:
            data_series = pandas.Series(data_0d)
            assert np.allclose(dmatrix(formula, data_series), expected)
            assert np.allclose(build_design_matrices([mat.design_info], data_series)[0], expected)