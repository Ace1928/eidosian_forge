import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def kp2_init(m, t):
    return m.A2 * pyo.exp(-m.E2 * 1000 / (m.R * m.T[t]))