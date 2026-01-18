import unittest
from pulp import GUROBI, LpProblem, LpVariable, const
@unittest.SkipTest
def test_gp_env_no_close(self):
    with gp.Env(params=self.env_options) as env:
        prob = generate_lp()
        solver = GUROBI(msg=False, env=env, **self.options)
        prob.solve(solver)
    self.assertRaises(gp.GurobiError, check_dummy_env)