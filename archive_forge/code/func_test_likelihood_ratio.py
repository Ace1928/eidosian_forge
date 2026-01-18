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
@unittest.skipIf(not graphics.imports_available, 'parmest.graphics imports are unavailable')
def test_likelihood_ratio(self):
    objval, thetavals = self.pest.theta_est()
    asym = np.arange(10, 30, 2)
    rate = np.arange(0, 1.5, 0.25)
    theta_vals = pd.DataFrame(list(product(asym, rate)), columns=self.pest.theta_names)
    obj_at_theta = self.pest.objective_at_theta(theta_vals)
    LR = self.pest.likelihood_ratio_test(obj_at_theta, objval, [0.8, 0.9, 1.0])
    self.assertTrue(set(LR.columns) >= set([0.8, 0.9, 1.0]))
    self.assertTrue(LR[0.8].sum() == 6)
    self.assertTrue(LR[0.9].sum() == 10)
    self.assertTrue(LR[1.0].sum() == 60)
    graphics.pairwise_plot(LR, thetavals, 0.8)