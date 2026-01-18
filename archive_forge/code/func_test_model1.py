import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
def test_model1(self):
    model = create_model1()
    nlp = PyomoNLP(model)
    solver = CyIpoptSolver(CyIpoptNLP(nlp))
    x, info = solver.solve(tee=False)
    x_sol = np.array([3.85958688, 4.67936007, 3.10358931])
    y_sol = np.array([-1.0, 53.90357665])
    self.assertTrue(np.allclose(x, x_sol, rtol=0.0001))
    nlp.set_primals(x)
    nlp.set_duals(y_sol)
    self.assertAlmostEqual(nlp.evaluate_objective(), -428.6362455416348, places=5)
    self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=0.0001))