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
def test_NA_action():
    initial_data = {'x': [1, 2, 3], 'c': ['c1', 'c2', 'c1']}

    def iter_maker():
        yield initial_data
    builder = design_matrix_builders([make_termlist('x', 'c')], iter_maker, 0)[0]
    mat = build_design_matrices([builder], {'x': [10.0, np.nan, 20.0], 'c': np.asarray(['c1', 'c2', None], dtype=object)})[0]
    assert mat.shape == (1, 3)
    assert np.array_equal(mat, [[1.0, 0.0, 10.0]])
    mat = build_design_matrices([builder], {'x': [10.0, np.nan, 20.0], 'c': np.asarray(['c1', 'c2', None], dtype=object)}, NA_action='drop')[0]
    assert mat.shape == (1, 3)
    assert np.array_equal(mat, [[1.0, 0.0, 10.0]])
    from patsy.missing import NAAction
    NA_action = NAAction(NA_types=[])
    mat = build_design_matrices([builder], {'x': [10.0, np.nan], 'c': np.asarray(['c1', 'c2'], dtype=object)}, NA_action=NA_action)[0]
    assert mat.shape == (2, 3)
    np.testing.assert_array_equal(mat, [[1.0, 0.0, 10.0], [0.0, 1.0, np.nan]])
    pytest.raises(PatsyError, build_design_matrices, [builder], {'x': [10.0, np.nan, 20.0], 'c': np.asarray(['c1', 'c2', None], dtype=object)}, NA_action='raise')