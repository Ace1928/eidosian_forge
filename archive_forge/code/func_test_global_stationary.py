import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from numpy.testing import assert_allclose, assert_raises
def test_global_stationary():
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), trend='c')
    intercept = 0
    phi = 0.5
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    check_initialization(mod, init, [0], np.diag([0]), np.eye(1) * sigma2 / (1 - phi ** 2))
    intercept = 1.2
    phi = 0.5
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    check_initialization(mod, init, [intercept / (1 - phi)], np.diag([0]), np.eye(1) * sigma2 / (1 - phi ** 2))
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0), trend='c')
    intercept = 0
    phi = [0.5, -0.2]
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    T = np.array([[0.5, 1], [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov = solve_discrete_lyapunov(T, Q)
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), desired_cov)
    intercept = 1.2
    phi = [0.5, -0.2]
    sigma2 = 2.0
    mod.update(np.r_[intercept, phi, sigma2])
    init = Initialization(mod.k_states, 'stationary')
    desired_intercept = np.linalg.inv(np.eye(2) - T).dot([intercept, 0])
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]), desired_cov)