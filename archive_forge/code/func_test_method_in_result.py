from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np
from scipy.optimize import root
@pytest.mark.parametrize('method', ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane'])
def test_method_in_result(self, method):

    def func(x):
        return x - 1
    res = root(func, x0=[1], method=method)
    assert res.method == method