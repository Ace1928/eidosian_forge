import unittest
from pulp import GUROBI, LpProblem, LpVariable, const
def check_dummy_env():
    with gp.Env(params={'OutputFlag': 0}):
        pass