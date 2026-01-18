import unittest
from pulp import GUROBI, LpProblem, LpVariable, const
def test_multiple_solves(self):
    solver = GUROBI(msg=False, manageEnv=True, **self.options)
    prob = generate_lp()
    prob.solve(solver)
    solver.close()
    check_dummy_env()
    solver2 = GUROBI(msg=False, manageEnv=True, **self.options)
    prob.solve(solver2)
    solver2.close()
    check_dummy_env()