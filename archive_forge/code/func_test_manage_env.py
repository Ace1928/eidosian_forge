import unittest
from pulp import GUROBI, LpProblem, LpVariable, const
def test_manage_env(self):
    solver = GUROBI(msg=False, manageEnv=True, **self.options)
    prob = generate_lp()
    prob.solve(solver)
    solver.close()
    check_dummy_env()