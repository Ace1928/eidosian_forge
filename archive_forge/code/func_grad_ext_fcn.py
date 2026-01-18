from pyomo.environ import ConcreteModel, Var, ExternalFunction, Objective
from pyomo.opt import SolverFactory
def grad_ext_fcn(args, fixed):
    a, b = args[:2]
    return [2 * a, 2 * b]