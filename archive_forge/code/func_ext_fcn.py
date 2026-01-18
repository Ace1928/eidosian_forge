from pyomo.environ import ConcreteModel, Var, ExternalFunction, Objective
from pyomo.opt import SolverFactory
def ext_fcn(a, b):
    return a ** 2 + b ** 2