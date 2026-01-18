from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def check_disjunction_transformation_block_structure(self, transBlock, pairs):
    self.assertIsInstance(transBlock, Block)
    disjBlock = transBlock.relaxedDisjuncts
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), len(pairs))
    bigm = TransformationFactory('gdp.bigm')
    for i, j in pairs:
        for comp in j:
            self.assertIs(bigm.get_transformed_constraints(comp)[0].parent_block(), disjBlock[i])