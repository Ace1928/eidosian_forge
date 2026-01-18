from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
def test_cov_scipy_curve_fit_comparison(self):
    """
        Scipy results differ in the 3rd decimal place from the paper. It is possible
        the paper used an alternative finite difference approximation for the Jacobian.
        """

    def model(t, asymptote, rate_constant):
        return asymptote * (1 - np.exp(-rate_constant * t))
    t = self.data['hour'].to_numpy()
    y = self.data['y'].to_numpy()
    theta_guess = np.array([15, 0.5])
    theta_hat, cov = scipy.optimize.curve_fit(model, t, y, p0=theta_guess)
    self.assertAlmostEqual(theta_hat[0], 19.1426, places=2)
    self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)
    self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)
    self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)
    self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)
    self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)