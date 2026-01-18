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
def test_named_expressions(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.E1 = Expression(expr=3 * (m.x * m.y + m.z))
    m.E2 = Expression(expr=m.z * m.y)
    m.E3 = Expression(expr=m.x * m.z + m.y)
    m.o1 = Objective(expr=m.E1 + m.E2)
    m.o2 = Objective(expr=m.E1 ** 2)
    m.c1 = Constraint(expr=m.E2 + 2 * m.E3 >= 0)
    m.c2 = Constraint(expr=pyo.inequality(0, m.E3 ** 2, 10))
    OUT = io.StringIO()
    nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True)
    self.assertEqual(*nl_diff("g3 1 1 0\t# problem unknown\n 3 2 2 1 0 \t# vars, constraints, objectives, ranges, eqns\n 2 2 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 3 3 3 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 6 6 \t# nonzeros in Jacobian, obj. gradient\n 2 1\t# max name lengths: constraints, variables\n 1 1 1 1 1\t# common exprs: b,c,o,c1,o1\nV3 0 0\t#nl(E1)\no2\t#*\nv0\t#x\nv1\t#y\nV4 0 0\t#E2\no2\t#*\nv2\t#z\nv1\t#y\nV5 0 0\t#nl(E3)\no2\t#*\nv0\t#x\nv2\t#z\nC0\t#c1\no0\t#+\nv4\t#E2\no2\t#*\nn2\nv5\t#nl(E3)\nV6 1 2\t#E3\n1 1\nv5\t#nl(E3)\nC1\t#c2\no5\t#^\nv6\t#E3\nn2\nO0 0\t#o1\no0\t#+\no2\t#*\nn3\nv3\t#nl(E1)\nv4\t#E2\nV7 1 4\t#E1\n2 3\no2\t#*\nn3\nv3\t#nl(E1)\nO1 0\t#o2\no5\t#^\nv7\t#E1\nn2\nx0\t# initial guess\nr\t#2 ranges (rhs's)\n2 0\t#c1\n0 0 10\t#c2\nb\t#3 bounds (on variables)\n3\t#x\n3\t#y\n3\t#z\nk2\t#intermediate Jacobian column lengths\n2\n4\nJ0 3\t#c1\n0 0\n1 2\n2 0\nJ1 3\t#c2\n0 0\n1 0\n2 0\nG0 3\t#o1\n0 0\n1 0\n2 3\nG1 3\t#o2\n0 0\n1 0\n2 0\n", OUT.getvalue()))