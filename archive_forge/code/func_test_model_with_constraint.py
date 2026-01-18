import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
def test_model_with_constraint(self):
    from pyomo.contrib.parmest.examples.rooney_biegler import rooney_biegler_with_constraint
    rooney_biegler_with_constraint.main()