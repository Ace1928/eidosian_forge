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
def test_model1_with_scaling(self):
    m = create_model1()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.o] = 1e-06
    m.scaling_factor[m.c] = 2.0
    m.scaling_factor[m.d] = 3.0
    m.scaling_factor[m.x[1]] = 4.0
    cynlp = CyIpoptNLP(PyomoNLP(m))
    options = {'nlp_scaling_method': 'user-scaling', 'output_file': '_cyipopt-scaling.log', 'file_print_level': 10, 'max_iter': 0}
    solver = CyIpoptSolver(cynlp, options=options)
    x, info = solver.solve()
    with open('_cyipopt-scaling.log', 'r') as fd:
        solver_trace = fd.read()
    cynlp.close()
    os.remove('_cyipopt-scaling.log')
    self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
    self.assertIn('output_file = _cyipopt-scaling.log', solver_trace)
    self.assertIn('objective scaling factor = 1e-06', solver_trace)
    self.assertIn('x scaling provided', solver_trace)
    self.assertIn('c scaling provided', solver_trace)
    self.assertIn('d scaling provided', solver_trace)
    self.assertIn('DenseVector "x scaling vector" with 3 elements:', solver_trace)
    self.assertIn('x scaling vector[    1]= 1.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    2]= 1.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    3]= 4.0000000000000000e+00', solver_trace)
    self.assertIn('DenseVector "c scaling vector" with 1 elements:', solver_trace)
    self.assertIn('c scaling vector[    1]= 2.0000000000000000e+00', solver_trace)
    self.assertIn('DenseVector "d scaling vector" with 1 elements:', solver_trace)
    self.assertIn('d scaling vector[    1]= 3.0000000000000000e+00', solver_trace)