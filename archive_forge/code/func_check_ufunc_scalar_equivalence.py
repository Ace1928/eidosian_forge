import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def check_ufunc_scalar_equivalence(op, arr1, arr2):
    scalar1 = arr1[()]
    scalar2 = arr2[()]
    assert isinstance(scalar1, np.generic)
    assert isinstance(scalar2, np.generic)
    if arr1.dtype.kind == 'c' or arr2.dtype.kind == 'c':
        comp_ops = {operator.ge, operator.gt, operator.le, operator.lt}
        if op in comp_ops and (np.isnan(scalar1) or np.isnan(scalar2)):
            pytest.xfail('complex comp ufuncs use sort-order, scalars do not.')
    if op == operator.pow and arr2.item() in [-1, 0, 0.5, 1, 2]:
        pytest.skip('array**2 can have incorrect/weird result dtype')
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('error', DeprecationWarning)
        try:
            res = op(arr1, arr2)
        except Exception as e:
            with pytest.raises(type(e)):
                op(scalar1, scalar2)
        else:
            scalar_res = op(scalar1, scalar2)
            assert_array_equal(scalar_res, res, strict=True)