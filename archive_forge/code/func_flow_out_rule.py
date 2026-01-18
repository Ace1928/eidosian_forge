import pyomo.environ as pyo
import pyomo.dae as dae
def flow_out_rule(m, t):
    return m.flow_out[t] - m.flow_const * pyo.sqrt(m.height[t]) == 0