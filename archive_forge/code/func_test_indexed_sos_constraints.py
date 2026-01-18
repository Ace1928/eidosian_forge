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
def test_indexed_sos_constraints(self):
    m = pyo.ConcreteModel()
    m.A = pyo.Set(initialize=[1])
    m.B = pyo.Set(initialize=[1, 2, 3])
    m.C = pyo.Set(initialize=[1])
    m.param_cx = pyo.Param(m.A, initialize={1: 1})
    m.param_cy = pyo.Param(m.B, initialize={1: 2, 2: 3, 3: 1})
    m.x = pyo.Var(m.A, domain=pyo.NonNegativeReals, bounds=(0, 40))
    m.y = pyo.Var(m.B, domain=pyo.NonNegativeIntegers)

    @m.Objective()
    def OBJ(m):
        return sum((m.param_cx[a] * m.x[a] for a in m.A)) + sum((m.param_cy[b] * m.y[b] for b in m.B))
    m.y[3].bounds = (2, 3)
    m.mysos = pyo.SOSConstraint(m.C, var=m.y, sos=1, index={1: [2, 3]}, weights={2: 25.0, 3: 18.0})
    OUT = io.StringIO()
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True)
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(*nl_diff("g3 1 1 0        # problem unknown\n 4 0 1 0 0      # vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0    # nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0    # network constraints: nonlinear, linear\n 0 0 0  # nonlinear vars in constraints, objectives, both\n 0 0 0 1        # linear network variables; functions; arith, flags\n 0 3 0 0 0      # discrete variables: binary, integer, nonlinear (b,c,o)\n 0 4    # nonzeros in Jacobian, obj. gradient\n 3 4    # max name lengths: constraints, variables\n 0 0 0 0 0      # common exprs: b,c,o,c1,o1\nS0 2 sosno\n2 1\n3 1\nS0 2 ref\n2 25.0\n3 18.0\nO0 0    #OBJ\nn0\nx0      # initial guess\nr       #0 ranges (rhs's)\nb       #4 bounds (on variables)\n0 0 40  #x[1]\n2 0     #y[1]\n2 0     #y[2]\n0 2 3   #y[3]\nk3      #intermediate Jacobian column lengths\n0\n0\n0\nG0 4    #OBJ\n0 1\n1 2\n2 3\n3 1\n", OUT.getvalue()))