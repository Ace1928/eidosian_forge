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
def test_duals_after_solve(self):
    m = pyo.ConcreteModel()
    m.p = pyo.Var(initialize=1)
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_models.OneOutput())
    m.con = pyo.Constraint(expr=4 * m.p - 2 * m.egb.outputs['o'] == 0)
    m.obj = pyo.Objective(expr=10 * m.p ** 2)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    solver = pyo.SolverFactory('cyipopt')
    status = solver.solve(m, tee=False)
    self.assertAlmostEqual(pyo.value(m.p), 10.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['u']), 4.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.outputs['o']), 20.0, places=3)
    self.assertAlmostEqual(pyo.value(m.dual[m.con]), 50.0, places=3)
    self.assertAlmostEqual(m.dual[m.egb]['egb.output_constraints[o]'], -100.0, places=3)
    self.assertAlmostEqual(pyo.value(m.ipopt_zL_out[m.egb.inputs['u']]), 500.0, places=3)
    self.assertAlmostEqual(pyo.value(m.ipopt_zU_out[m.egb.inputs['u']]), 0.0, places=3)
    del m.obj
    m.obj = pyo.Objective(expr=-10 * m.p ** 2)
    status = solver.solve(m, tee=False)
    self.assertAlmostEqual(pyo.value(m.p), 25.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['u']), 10.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.outputs['o']), 50.0, places=3)
    self.assertAlmostEqual(pyo.value(m.dual[m.con]), -125.0, places=3)
    self.assertAlmostEqual(m.dual[m.egb]['egb.output_constraints[o]'], 250.0, places=3)
    self.assertAlmostEqual(pyo.value(m.ipopt_zL_out[m.egb.inputs['u']]), 0.0, places=3)
    self.assertAlmostEqual(pyo.value(m.ipopt_zU_out[m.egb.inputs['u']]), -1250.0, places=3)
    m = pyo.ConcreteModel()
    m.p = pyo.Var(initialize=1)
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_models.OneOutputOneEquality())
    m.con = pyo.Constraint(expr=4 * m.p - 2 * m.egb.outputs['o'] == 0)
    m.obj = pyo.Objective(expr=10 * m.p ** 2)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    solver = pyo.SolverFactory('cyipopt')
    status = solver.solve(m, tee=False)
    self.assertAlmostEqual(pyo.value(m.p), 2.5, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.inputs['u']), 1.0, places=3)
    self.assertAlmostEqual(pyo.value(m.egb.outputs['o']), 5.0, places=3)
    self.assertAlmostEqual(pyo.value(m.dual[m.con]), 12.5, places=3)
    self.assertAlmostEqual(m.dual[m.egb]['egb.output_constraints[o]'], -25.0, places=3)
    self.assertAlmostEqual(m.dual[m.egb]['egb.u2_con'], 62.5, places=3)