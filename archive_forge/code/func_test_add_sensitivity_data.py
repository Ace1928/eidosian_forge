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
def test_add_sensitivity_data(self):
    model = make_indexed_model()
    sens = SensitivityInterface(model, clone_model=False)
    sens._add_data_block()
    param_list = [model.x, model.eta]
    with self.assertRaises(ValueError) as exc:
        sens._add_sensitivity_data(param_list)
    self.assertIn('variables must be fixed', str(exc.exception))
    sens.model_instance.x.fix()
    param_list = [model.x, model.x[1], model.eta, model.eta[1]]
    sens._add_sensitivity_data(param_list)
    block_param_list = list(sens.block.component_data_objects(Param))
    block_var_list = list(sens.block.component_data_objects(Var))
    self.assertEqual(len(block_param_list), 4)
    self.assertEqual(len(block_var_list), 3)
    self.assertEqual(len(sens.block._sens_data_list), 7)
    pred_sens_data_list = [(model.x[1], Param, 0, 1), (model.x[2], Param, 0, 2), (model.x[3], Param, 0, 3), (model.x[1], Param, 1, _NotAnIndex), (Var, model.eta[1], 2, 1), (Var, model.eta[2], 2, 2), (Var, model.eta[1], 3, _NotAnIndex)]
    for data, pred in zip(sens.block._sens_data_list, pred_sens_data_list):
        if isinstance(pred[0], ComponentData):
            self.assertIs(data[0], pred[0])
            self.assertIs(data[1].ctype, pred[1])
            name = data[0].parent_component().local_name
            self.assertTrue(data[1].parent_component().local_name.startswith(name))
        else:
            self.assertIs(data[0].ctype, pred[0])
            self.assertIs(data[1], pred[1])
            name = data[1].parent_component().local_name
            self.assertTrue(data[0].parent_component().local_name.startswith(name))
        self.assertEqual(data[2], pred[2])
        self.assertEqual(data[3], pred[3])