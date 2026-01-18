from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_advanced2(self) -> None:
    """Test code from the advanced section of the tutorial.
        """
    x = cvx.Variable()
    prob = cvx.Problem(cvx.Minimize(cvx.square(x)), [x == 2])
    data, chain, inverse = prob.get_problem_data(cvx.ECOS)
    if cvx.CVXOPT in cvx.installed_solvers():
        data, chain, inverse = prob.get_problem_data(cvx.CVXOPT)
    data, chain, inverse = prob.get_problem_data(cvx.SCS)
    import ecos
    data, chain, inverse = prob.get_problem_data(cvx.ECOS)
    solution = ecos.solve(data['c'], data['G'], data['h'], ecos_conif.dims_to_solver_dict(data['dims']), data['A'], data['b'])
    prob.unpack_results(solution, chain, inverse)