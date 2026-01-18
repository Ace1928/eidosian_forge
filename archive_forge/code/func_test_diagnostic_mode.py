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
def test_diagnostic_mode(self):
    self.pest.diagnostic_mode = True
    objval, thetavals = self.pest.theta_est()
    asym = np.arange(10, 30, 2)
    rate = np.arange(0, 1.5, 0.25)
    theta_vals = pd.DataFrame(list(product(asym, rate)), columns=self.pest.theta_names)
    obj_at_theta = self.pest.objective_at_theta(theta_vals)
    self.pest.diagnostic_mode = False