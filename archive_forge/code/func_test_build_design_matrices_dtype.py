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
def test_build_design_matrices_dtype():
    data = {'x': [1, 2, 3]}

    def iter_maker():
        yield data
    builder = design_matrix_builders([make_termlist('x')], iter_maker, 0)[0]
    mat = build_design_matrices([builder], data)[0]
    assert mat.dtype == np.dtype(np.float64)
    mat = build_design_matrices([builder], data, dtype=np.float32)[0]
    assert mat.dtype == np.dtype(np.float32)
    if hasattr(np, 'float128'):
        mat = build_design_matrices([builder], data, dtype=np.float128)[0]
        assert mat.dtype == np.dtype(np.float128)