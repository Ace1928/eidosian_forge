import pyomo.environ as pyo
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def create_pyomo_reactor_model():
    m = pyo.ConcreteModel()
    k1 = 5.0 / 6.0
    k2 = 5.0 / 3.0
    k3 = 1.0 / 6000.0
    m.sv = pyo.Var(initialize=1.0, bounds=(0, None))
    m.caf = pyo.Var(initialize=1.0)
    m.ca = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cb = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cc = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cd = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cb_ratio = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=m.cb_ratio, sense=pyo.maximize)
    m.ca_bal = pyo.Constraint(expr=0 == m.sv * m.caf - m.sv * m.ca - k1 * m.ca - 2.0 * k3 * m.ca ** 2.0)
    m.cb_bal = pyo.Constraint(expr=0 == -m.sv * m.cb + k1 * m.ca - k2 * m.cb)
    m.cc_bal = pyo.Constraint(expr=0 == -m.sv * m.cc + k2 * m.cb)
    m.cd_bal = pyo.Constraint(expr=0 == -m.sv * m.cd + k3 * m.ca ** 2.0)
    m.cb_ratio_con = pyo.Constraint(expr=m.cb / (m.ca + m.cc + m.cd) - m.cb_ratio == 0)
    m.cafcon = pyo.Constraint(expr=m.caf == 10000)
    return m