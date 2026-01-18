import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
@unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
def test_calculateFeasibility(self):
    self.interface.replaceExternalFunctionsWithVariables()
    self.interface.createConstraints()
    self.interface.model.x[0] = 2.0
    self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
    self.interface.data.basis_model_output[:] = 0
    self.interface.data.grad_basis_model_output[...] = 0
    self.interface.data.truth_model_output[:] = 0
    self.interface.data.grad_truth_model_output[...] = 0
    self.interface.data.value_of_ef_inputs[...] = 0
    feasibility = self.interface.calculateFeasibility()
    self.assertEqual(feasibility, 0)
    self.interface.updateSurrogateModel()
    feasibility = self.interface.calculateFeasibility()
    self.assertEqual(feasibility, 0)
    self.interface.data.basis_constraint.activate()
    objective, step_norm, feasibility = self.interface.solveModel()
    self.assertEqual(feasibility, 0.09569982275514467)
    self.interface.data.basis_constraint.deactivate()