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
def test_data_independent_builder():
    data = {'x': [1, 2, 3]}

    def iter_maker():
        yield data
    null_builder = design_matrix_builders([make_termlist()], iter_maker, 0)[0]
    pytest.raises(PatsyError, build_design_matrices, [null_builder], data)
    intercept_builder = design_matrix_builders([make_termlist([])], iter_maker, eval_env=0)[0]
    pytest.raises(PatsyError, build_design_matrices, [intercept_builder], data)
    pytest.raises(PatsyError, build_design_matrices, [null_builder, intercept_builder], data)
    if have_pandas:
        int_m, null_m = build_design_matrices([intercept_builder, null_builder], pandas.DataFrame(data))
        assert np.allclose(int_m, [[1], [1], [1]])
        assert null_m.shape == (3, 0)
    x_termlist = make_termlist(['x'])
    builders = design_matrix_builders([x_termlist, make_termlist()], iter_maker, eval_env=0)
    x_m, null_m = build_design_matrices(builders, data)
    assert np.allclose(x_m, [[1], [2], [3]])
    assert null_m.shape == (3, 0)
    builders = design_matrix_builders([x_termlist, make_termlist([])], iter_maker, eval_env=0)
    x_m, null_m = build_design_matrices(builders, data)
    x_m, intercept_m = build_design_matrices(builders, data)
    assert np.allclose(x_m, [[1], [2], [3]])
    assert np.allclose(intercept_m, [[1], [1], [1]])