import pyomo.environ as pyo
import pyomo.dae as dae
def mbal(m, i):
    if i == 0:
        return pyo.Constraint.Skip
    else:
        return m.rho[i - 1] * m.F[i - 1] - m.rho[i] * m.F[i] == 0