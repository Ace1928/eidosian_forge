import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
@unittest.pytest.mark.expensive
def test_likelihood_ratio_example(self):
    from pyomo.contrib.parmest.examples.reactor_design import likelihood_ratio_example
    likelihood_ratio_example.main()