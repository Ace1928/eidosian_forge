import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_chain_data_for_non_dpp_problem_evals_params(self) -> None:
    x = cp.Parameter()
    x.value = 5
    y = cp.Variable()
    problem = cp.Problem(cp.Minimize(x * x), [x == y])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, chain, _ = problem.get_problem_data(cp.SCS)
    self.assertFalse(problem.is_dpp())
    self.assertTrue(cp.reductions.eval_params.EvalParams in [type(r) for r in chain.reductions])