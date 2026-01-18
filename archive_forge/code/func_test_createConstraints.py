import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
def test_createConstraints(self):
    self.interface.replaceExternalFunctionsWithVariables()
    self.interface.createConstraints()
    self.assertFalse(self.interface.data.basis_constraint.active)
    self.assertFalse(self.interface.data.sm_constraint_basis.active)
    self.assertEqual(len(self.interface.data.basis_constraint), 1)
    self.assertEqual(len(self.interface.data.sm_constraint_basis), 1)
    self.assertEqual(list(self.interface.data.basis_constraint.keys()), [1])
    cs = ComponentSet(identify_variables(self.interface.data.basis_constraint[1].expr))
    self.assertEqual(len(cs), 1)
    self.assertIn(self.interface.data.ef_outputs[1], cs)
    cs = ComponentSet(identify_variables(self.interface.data.sm_constraint_basis[1].expr))
    self.assertEqual(len(cs), 3)
    self.assertIn(self.interface.model.x[0], cs)
    self.assertIn(self.interface.model.x[1], cs)
    self.assertIn(self.interface.data.ef_outputs[1], cs)