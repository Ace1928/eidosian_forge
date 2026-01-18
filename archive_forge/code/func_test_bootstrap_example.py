import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
@unittest.skipUnless(seaborn_available, 'test requires seaborn')
def test_bootstrap_example(self):
    from pyomo.contrib.parmest.examples.reactor_design import bootstrap_example
    bootstrap_example.main()