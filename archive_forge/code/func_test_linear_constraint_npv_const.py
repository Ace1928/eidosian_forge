import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
def test_linear_constraint_npv_const(self):
    m = ConcreteModel()
    m.x = Var([1, 2])
    m.p = Param(initialize=5, mutable=True)
    m.o = Objective(expr=1)
    m.c = Constraint(expr=LinearExpression([m.p ** 2, 5 * m.x[1], 10 * m.x[2]]) <= 0)
    OUT = io.StringIO()
    nl_writer.NLWriter().write(m, OUT)
    self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 2 1 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 0 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 2 0 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nC0\nn0\nO0 0\nn1.0\nx0\nr\n1 -25\nb\n3\n3\nk1\n1\nJ0 2\n0 5\n1 10\n', OUT.getvalue()))