import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.core.base.component import ComponentData
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters
from pyomo.contrib.sensitivity_toolbox.sens import (
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_example
from pyomo.opt import SolverFactory
from pyomo.common.dependencies import (
from pyomo.common.dependencies import scipy_available
def test_add_data_block(self):
    model = param_example.create_model()
    sens = SensitivityInterface(model, clone_model=False)
    block = sens._add_data_block()
    self.assertIs(sens.block.parent_block(), sens.model_instance)
    self.assertIs(sens.block.ctype, Block)
    self.assertEqual(sens.block.local_name, sens.get_default_block_name())
    with self.assertRaises(RuntimeError) as ex:
        sens._add_data_block()
    self.assertIn('Cannot add component', str(ex.exception))
    new_block = sens._add_data_block(existing_block=block)
    self.assertIsNot(block, new_block)
    new_block._has_replaced_expressions = True
    with self.assertRaises(RuntimeError) as ex:
        sens._add_data_block(existing_block=new_block)
    self.assertIn('Re-using sensitivity interface', str(ex.exception))