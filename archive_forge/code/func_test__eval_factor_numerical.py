import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def test__eval_factor_numerical():
    import pytest
    naa = NAAction()
    f = _MockFactor()
    fi1 = FactorInfo(f, 'numerical', {}, num_columns=1, categories=None)
    assert fi1.factor is f
    eval123, is_NA = _eval_factor(fi1, {'mock': [1, 2, 3]}, naa)
    assert eval123.shape == (3, 1)
    assert np.all(eval123 == [[1], [2], [3]])
    assert is_NA.shape == (3,)
    assert np.all(~is_NA)
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': [[[1]]]}, naa)
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': [[1, 2]]}, naa)
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': ['a', 'b']}, naa)
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': [True, False]}, naa)
    fi2 = FactorInfo(_MockFactor(), 'numerical', {}, num_columns=2, categories=None)
    eval123321, is_NA = _eval_factor(fi2, {'mock': [[1, 3], [2, 2], [3, 1]]}, naa)
    assert eval123321.shape == (3, 2)
    assert np.all(eval123321 == [[1, 3], [2, 2], [3, 1]])
    assert is_NA.shape == (3,)
    assert np.all(~is_NA)
    pytest.raises(PatsyError, _eval_factor, fi2, {'mock': [1, 2, 3]}, naa)
    pytest.raises(PatsyError, _eval_factor, fi2, {'mock': [[1, 2, 3]]}, naa)
    ev_nan, is_NA = _eval_factor(fi1, {'mock': [1, 2, np.nan]}, NAAction(NA_types=['NaN']))
    assert np.array_equal(is_NA, [False, False, True])
    ev_nan, is_NA = _eval_factor(fi1, {'mock': [1, 2, np.nan]}, NAAction(NA_types=[]))
    assert np.array_equal(is_NA, [False, False, False])
    if have_pandas:
        eval_ser, _ = _eval_factor(fi1, {'mock': pandas.Series([1, 2, 3], index=[10, 20, 30])}, naa)
        assert isinstance(eval_ser, pandas.DataFrame)
        assert np.array_equal(eval_ser, [[1], [2], [3]])
        assert np.array_equal(eval_ser.index, [10, 20, 30])
        eval_df1, _ = _eval_factor(fi1, {'mock': pandas.DataFrame([[2], [1], [3]], index=[20, 10, 30])}, naa)
        assert isinstance(eval_df1, pandas.DataFrame)
        assert np.array_equal(eval_df1, [[2], [1], [3]])
        assert np.array_equal(eval_df1.index, [20, 10, 30])
        eval_df2, _ = _eval_factor(fi2, {'mock': pandas.DataFrame([[2, 3], [1, 4], [3, -1]], index=[20, 30, 10])}, naa)
        assert isinstance(eval_df2, pandas.DataFrame)
        assert np.array_equal(eval_df2, [[2, 3], [1, 4], [3, -1]])
        assert np.array_equal(eval_df2.index, [20, 30, 10])
        pytest.raises(PatsyError, _eval_factor, fi2, {'mock': pandas.Series([1, 2, 3], index=[10, 20, 30])}, naa)
        pytest.raises(PatsyError, _eval_factor, fi1, {'mock': pandas.DataFrame([[2, 3], [1, 4], [3, -1]], index=[20, 30, 10])}, naa)