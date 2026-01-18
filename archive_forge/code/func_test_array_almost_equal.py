import pytest
import numpy as np
from ase.calculators.calculator import equal
@pytest.mark.parametrize('a', arrays([2, 2], float))
@pytest.mark.parametrize('b', arrays(np.array([2, 2]) + 1.9e-08, float))
@pytest.mark.parametrize('rtol,atol', [[None, 2e-08], [0, 2e-08], [1e-08, None], [1e-08, 0], [5e-09, 1e-08]])
def test_array_almost_equal(a, b, rtol, atol):
    assert a is not b
    assert equal(a, b, rtol=rtol, atol=atol)
    assert equal({'size': a, 'gamma': True}, {'size': b, 'gamma': True}, rtol=rtol, atol=atol)