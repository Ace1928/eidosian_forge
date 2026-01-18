from pyomo.common.dependencies import pandas as pd, pandas_available
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.opt import SolverFactory
@unittest.pytest.mark.expensive
def test_convert_param_to_var(self):
    from pyomo.contrib.parmest.examples.reactor_design.reactor_design import reactor_design_model
    data = pd.DataFrame(data=[[1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5], [1.1, 10000, 3535.1, 1064.8, 1613.3, 1893.4], [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8]], columns=['sv', 'caf', 'ca', 'cb', 'cc', 'cd'])
    theta_names = ['k1', 'k2', 'k3']
    instance = reactor_design_model(data.loc[0])
    solver = pyo.SolverFactory('ipopt')
    solver.solve(instance)
    instance_vars = parmest.utils.convert_params_to_vars(instance, theta_names, fix_vars=True)
    solver.solve(instance_vars)
    assert instance.k1() == instance_vars.k1()
    assert instance.k2() == instance_vars.k2()
    assert instance.k3() == instance_vars.k3()