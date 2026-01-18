import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from numpy.testing import assert_allclose, assert_raises
def test_global_known():
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    init = Initialization(mod.k_states, 'known', constant=[1.5])
    check_initialization(mod, init, [1.5], np.diag([0]), np.diag([0]))
    init = Initialization(mod.k_states, 'known', stationary_cov=np.diag([1]))
    check_initialization(mod, init, [0], np.diag([0]), np.diag([1]))
    init = Initialization(mod.k_states, 'known', constant=[1.5], stationary_cov=np.diag([1]))
    check_initialization(mod, init, [1.5], np.diag([0]), np.diag([1]))
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    init = Initialization(mod.k_states, 'known', constant=[1.5, -0.2])
    check_initialization(mod, init, [1.5, -0.2], np.diag([0, 0]), np.diag([0, 0]))
    init = Initialization(mod.k_states, 'known', stationary_cov=np.diag([1, 4.2]))
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.diag([1, 4.2]))
    init = Initialization(mod.k_states, 'known', constant=[1.5, -0.2], stationary_cov=np.diag([1, 4.2]))
    check_initialization(mod, init, [1.5, -0.2], np.diag([0, 0]), np.diag([1, 4.2]))