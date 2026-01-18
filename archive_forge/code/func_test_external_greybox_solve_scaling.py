import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
@unittest.skipIf(not cyipopt_available, 'CyIpopt needed to run tests with solve')
def test_external_greybox_solve_scaling(self):
    m = pyo.ConcreteModel()
    m.mu = pyo.Var(bounds=(0, None), initialize=1)
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_models.PressureDropTwoEqualitiesTwoOutputsScaleBoth())
    m.ccon = pyo.Constraint(expr=m.egb.inputs['c'] == 128 / (3.14 * 0.0001) * m.mu * m.egb.inputs['F'])
    m.pcon = pyo.Constraint(expr=m.egb.inputs['Pin'] - m.egb.outputs['Pout'] <= 72)
    m.pincon = pyo.Constraint(expr=m.egb.inputs['Pin'] == 100.0)
    m.egb.inputs['Pin'].value = 100
    m.egb.inputs['Pin'].setlb(50)
    m.egb.inputs['Pin'].setub(150)
    m.egb.inputs['c'].value = 2
    m.egb.inputs['c'].setlb(1)
    m.egb.inputs['c'].setub(5)
    m.egb.inputs['F'].value = 3
    m.egb.inputs['F'].setlb(1)
    m.egb.inputs['F'].setub(5)
    m.egb.inputs['P1'].value = 80
    m.egb.inputs['P1'].setlb(10)
    m.egb.inputs['P1'].setub(90)
    m.egb.inputs['P3'].value = 70
    m.egb.inputs['P3'].setlb(20)
    m.egb.inputs['P3'].setub(80)
    m.egb.outputs['P2'].value = 75
    m.egb.outputs['P2'].setlb(15)
    m.egb.outputs['P2'].setub(85)
    m.egb.outputs['Pout'].value = 50
    m.egb.outputs['Pout'].setlb(10)
    m.egb.outputs['Pout'].setub(70)
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2 + (m.egb.inputs['F'] - 3) ** 2)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.obj] = 0.1
    m.scaling_factor[m.egb.inputs['Pin']] = 1.1
    m.scaling_factor[m.egb.inputs['c']] = 1.2
    m.scaling_factor[m.egb.inputs['F']] = 1.3
    m.scaling_factor[m.egb.inputs['P3']] = 1.5
    m.scaling_factor[m.egb.outputs['P2']] = 1.6
    m.scaling_factor[m.egb.outputs['Pout']] = 1.7
    m.scaling_factor[m.mu] = 1.9
    m.scaling_factor[m.pincon] = 2.2
    solver = pyo.SolverFactory('cyipopt')
    solver.config.options = {'hessian_approximation': 'limited-memory', 'nlp_scaling_method': 'user-scaling', 'output_file': '_cyipopt-external-greybox-scaling.log', 'file_print_level': 10, 'max_iter': 0}
    status = solver.solve(m, tee=False)
    with open('_cyipopt-external-greybox-scaling.log', 'r') as fd:
        solver_trace = fd.read()
    os.remove('_cyipopt-external-greybox-scaling.log')
    self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
    self.assertIn('output_file = _cyipopt-external-greybox-scaling.log', solver_trace)
    self.assertIn('objective scaling factor = 0.1', solver_trace)
    self.assertIn('x scaling provided', solver_trace)
    self.assertIn('c scaling provided', solver_trace)
    self.assertIn('d scaling provided', solver_trace)
    self.assertIn('DenseVector "x scaling vector" with 8 elements:', solver_trace)
    self.assertIn('x scaling vector[    1]= 1.3000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    8]= 1.8999999999999999e+00', solver_trace)
    self.assertIn('x scaling vector[    7]= 1.7000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    4]= 1.1000000000000001e+00', solver_trace)
    self.assertIn('x scaling vector[    5]= 1.2000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    2]= 1.0000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    3]= 1.5000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    6]= 1.6000000000000001e+00', solver_trace)
    self.assertIn('DenseVector "c scaling vector" with 6 elements:', solver_trace)
    self.assertIn('c scaling vector[    1]= 1.0000000000000000e+00', solver_trace)
    self.assertIn('c scaling vector[    2]= 2.2000000000000002e+00', solver_trace)
    self.assertIn('c scaling vector[    3]= 3.1000000000000001e+00', solver_trace)
    self.assertIn('c scaling vector[    4]= 3.2000000000000002e+00', solver_trace)
    self.assertIn('c scaling vector[    5]= 4.0999999999999996e+00', solver_trace)
    self.assertIn('c scaling vector[    6]= 4.2000000000000002e+00', solver_trace)
    self.assertIn('DenseVector "d scaling vector" with 1 elements:', solver_trace)
    self.assertIn('d scaling vector[    1]= 1.0000000000000000e+00', solver_trace)