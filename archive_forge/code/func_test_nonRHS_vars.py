import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
@unittest.skipIf(not casadi_available, 'casadi not available')
def test_nonRHS_vars(self):
    m = self.m
    m.v2 = Var(m.t)
    m.dv2 = DerivativeVar(m.v2)
    m.p = Param(initialize=5)
    t = IndexTemplate(m.t)

    def _con(m, t):
        return m.dv2[t] == 10 + m.p
    m.con = Constraint(m.t, rule=_con)
    mysim = Simulator(m, package='casadi')
    self.assertEqual(len(mysim._templatemap), 1)
    self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v2[t]))
    m.del_component('con')