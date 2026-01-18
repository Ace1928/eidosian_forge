import unittest
from pulp import GUROBI, LpProblem, LpVariable, const
def generate_lp() -> LpProblem:
    prob = LpProblem('test', const.LpMaximize)
    x = LpVariable('x', 0, 1)
    y = LpVariable('y', 0, 1)
    z = LpVariable('z', 0, 1)
    prob += (x + y + z, 'obj')
    prob += (x + y + z <= 1, 'c1')
    return prob