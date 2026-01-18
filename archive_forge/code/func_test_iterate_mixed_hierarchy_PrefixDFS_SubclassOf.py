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
def test_iterate_mixed_hierarchy_PrefixDFS_SubclassOf(self):
    HM = MixedHierarchicalModel()
    m = HM.model
    result = [x.name for x in m.block_data_objects(descent_order=TraversalStrategy.PrefixDepthFirstSearch, descend_into=SubclassOf(Block))]
    self.assertEqual(HM.PrefixDFS_both, result)
    result = [x.name for x in m.component_objects(ctype=Block, descent_order=TraversalStrategy.PrefixDepthFirstSearch, descend_into=SubclassOf(Block))]
    self.assertEqual(HM.PrefixDFS_block_subclass, result)
    result = [x.name for x in m.component_objects(ctype=Block, descent_order=TraversalStrategy.PrefixDepthFirstSearch, descend_into=SubclassOf(Var, Block))]
    self.assertEqual(HM.PrefixDFS_block_subclass, result)