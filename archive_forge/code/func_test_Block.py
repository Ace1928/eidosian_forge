from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_Block(self):
    self.generator_runner(Block)
    model = self.generate_model()
    self.assertEqual([id(comp) for comp in model.block_data_objects(sort=SortComponents.deterministic)], [id(comp) for comp in [model] + model.component_data_lists[Block]])
    self.assertEqual(sorted([id(comp) for comp in model.block_data_objects(sort=False)]), sorted([id(comp) for comp in [model] + model.component_data_lists[Block]]))