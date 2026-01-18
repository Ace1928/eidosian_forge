import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
@unittest.pytest.mark.expensive
def test_leaveNout_example(self):
    from pyomo.contrib.parmest.examples.reactor_design import leaveNout_example
    leaveNout_example.main()