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
def test_dataformats(self):
    obj1, theta1 = self.pest_df.theta_est()
    obj2, theta2 = self.pest_dict.theta_est()
    self.assertAlmostEqual(obj1, obj2, places=6)
    self.assertAlmostEqual(theta1['k1'], theta2['k1'], places=6)
    self.assertAlmostEqual(theta1['k2'], theta2['k2'], places=6)