import pyomo.environ as pyo
import pyomo.dae as dae
def diff_eqn_rule(m, t):
    return m.area * m.dhdt[t] - (m.flow_in[t] - m.flow_out[t]) == 0