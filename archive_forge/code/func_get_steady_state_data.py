import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.model import (
def get_steady_state_data(target, tee=False):
    m = create_instance(dynamic=False)
    interface = mpc.DynamicModelInterface(m, m.time)
    var_set, tr_cost = interface.get_penalty_from_target(target)
    m.target_set = var_set
    m.tracking_cost = tr_cost
    m.objective = pyo.Objective(expr=sum(m.tracking_cost[:, 0]))
    m.flow_in[:].unfix()
    solver = pyo.SolverFactory('ipopt')
    solver.solve(m, tee=tee)
    return interface.get_data_at_time(0)