import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
@unittest.skipUnless(matplotlib_available, 'test requires matplotlib')
def test_datarec_example(self):
    from pyomo.contrib.parmest.examples.reactor_design import datarec_example
    datarec_example.main()