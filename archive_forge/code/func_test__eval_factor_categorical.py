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
def test__eval_factor_categorical():
    import pytest
    from patsy.categorical import C
    naa = NAAction()
    f = _MockFactor()
    fi1 = FactorInfo(f, 'categorical', {}, num_columns=None, categories=('a', 'b'))
    assert fi1.factor is f
    cat1, _ = _eval_factor(fi1, {'mock': ['b', 'a', 'b']}, naa)
    assert cat1.shape == (3,)
    assert np.all(cat1 == [1, 0, 1])
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': ['c']}, naa)
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': C(['a', 'c'])}, naa)
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': C(['a', 'b'], levels=['b', 'a'])}, naa)
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': [1, 0, 1]}, naa)
    bad_cat = np.asarray(['b', 'a', 'a', 'b'])
    bad_cat.resize((2, 2))
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': bad_cat}, naa)
    cat1_NA, is_NA = _eval_factor(fi1, {'mock': ['a', None, 'b']}, NAAction(NA_types=['None']))
    assert np.array_equal(is_NA, [False, True, False])
    assert np.array_equal(cat1_NA, [0, -1, 1])
    pytest.raises(PatsyError, _eval_factor, fi1, {'mock': ['a', None, 'b']}, NAAction(NA_types=[]))
    fi2 = FactorInfo(_MockFactor(), 'categorical', {}, num_columns=None, categories=[False, True])
    cat2, _ = _eval_factor(fi2, {'mock': [True, False, False, True]}, naa)
    assert cat2.shape == (4,)
    assert np.all(cat2 == [1, 0, 0, 1])
    if have_pandas:
        s = pandas.Series(['b', 'a'], index=[10, 20])
        cat_s, _ = _eval_factor(fi1, {'mock': s}, naa)
        assert isinstance(cat_s, pandas.Series)
        assert np.array_equal(cat_s, [1, 0])
        assert np.array_equal(cat_s.index, [10, 20])
        sbool = pandas.Series([True, False], index=[11, 21])
        cat_sbool, _ = _eval_factor(fi2, {'mock': sbool}, naa)
        assert isinstance(cat_sbool, pandas.Series)
        assert np.array_equal(cat_sbool, [1, 0])
        assert np.array_equal(cat_sbool.index, [11, 21])