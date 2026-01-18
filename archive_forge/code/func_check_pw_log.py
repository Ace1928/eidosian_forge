import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
def check_pw_log(self, m):
    z = m.pw_log.get_transformation_var(m.log_expr)
    self.assertIsInstance(z, Var)
    log_block = z.parent_block()
    ct.check_trans_block_structure(self, log_block)
    self.assertEqual(len(log_block.disjuncts), 3)
    disjuncts_dict = {log_block.disjuncts[0]: ((1, 3), m.f1), log_block.disjuncts[1]: ((3, 6), m.f2), log_block.disjuncts[2]: ((6, 10), m.f3)}
    for d, (pts, f) in disjuncts_dict.items():
        self.check_log_disjunct(d, pts, f, log_block.substitute_var, m.x)
    self.assertIsInstance(log_block.pick_a_piece, Disjunction)
    self.assertEqual(len(log_block.pick_a_piece.disjuncts), 3)
    for i in range(2):
        self.assertIs(log_block.pick_a_piece.disjuncts[i], log_block.disjuncts[i])
    self.assertIs(m.obj.expr.expr, log_block.substitute_var)