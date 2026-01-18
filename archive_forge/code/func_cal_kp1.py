import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def cal_kp1(m, t):
    """
            Create the perturbation parameter sets
            m: model
            t: time
            """
    return m.kp1[t] == m.A1 * pyo.exp(-m.E1 * 1000 / (m.R * m.T[t]))