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
def test_NA_drop_preserves_levels():
    data = {'x': [1.0, np.nan, 3.0], 'c': ['c1', 'c2', 'c3']}

    def iter_maker():
        yield data
    design_info = design_matrix_builders([make_termlist('x', 'c')], iter_maker, 0)[0]
    assert design_info.column_names == ['c[c1]', 'c[c2]', 'c[c3]', 'x']
    mat, = build_design_matrices([design_info], data)
    assert mat.shape == (2, 4)
    assert np.array_equal(mat, [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 3.0]])