import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
def test_dogleg_user_warning(self):
    with pytest.warns(RuntimeWarning, match='Maximum number of iterations'):
        minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess, method='dogleg', options={'disp': True, 'maxiter': 1})