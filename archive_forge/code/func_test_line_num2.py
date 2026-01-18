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
def test_line_num2(self):
    """
        It tests an exception error when file does not include target
        """
    import os
    file_name = 'test_col.col'
    with open(file_name, 'w') as file:
        file.write('var1\n')
        file.write('var3\n')
    with self.assertRaises(Exception) as context:
        i = line_num(file_name, 'var2')
    self.assertTrue('test_col.col does not include var2' in str(context.exception))