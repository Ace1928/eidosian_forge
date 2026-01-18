import pyomo.contrib.piecewise.tests.models as models
from pyomo.core import Var
from pyomo.core.base import TransformationFactory
from pyomo.environ import value
from pyomo.gdp import Disjunct, Disjunction
def check_trans_block_structure(test, block):
    test.assertEqual(len(block.component_map(Disjunct)), 1)
    test.assertEqual(len(block.component_map(Disjunction)), 1)
    test.assertEqual(len(block.component_map(Var)), 1)
    test.assertIsInstance(block.substitute_var, Var)