import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
def test_initializeInterface(self):
    self.assertEqual(self.m, self.interface.original_model)
    self.assertEqual(self.config, self.interface.config)
    self.assertEqual(self.interface.basis_expression_rule, self.ext_fcn_surrogate_map_rule)
    self.assertEqual('ipopt', self.interface.solver.name)