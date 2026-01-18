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
@unittest.skipIf(not scipy_available, 'Scipy is not available')
def test_non_supported_multi_index(self):
    m = self.m
    m.v2 = Var(m.t, m.s)
    m.v3 = Var(m.s, m.t)
    m.dv2 = DerivativeVar(m.v2)
    m.dv3 = DerivativeVar(m.v3)

    def _diffeq(m, t, s):
        return m.dv2[t, s] == m.v2[t, s] ** 2 + m.v2[t, s]
    m.con1 = Constraint(m.t, m.s, rule=_diffeq)
    m.con2 = Constraint(m.t, m.s, rule=_diffeq)
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m.del_component('con1')
    m.del_component('con2')
    m.del_component('con1_index')
    m.del_component('con2_index')

    def _diffeq(m, s, t):
        return m.dv3[s, t] == m.v3[s, t] ** 2 + m.v3[s, t]
    m.con1 = Constraint(m.s, m.t, rule=_diffeq)
    m.con2 = Constraint(m.s, m.t, rule=_diffeq)
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m.del_component('con1')
    m.del_component('con2')
    m.del_component('con1_index')
    m.del_component('con2_index')

    def _diffeq(m, t, s):
        return m.dv2[t, s] == m.dv2[t, s] + m.v2[t, s] ** 2
    m.con1 = Constraint(m.t, m.s, rule=_diffeq)
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m.del_component('con1')
    m.del_component('con1_index')

    def _diffeq(m, s, t):
        return m.dv3[s, t] == m.dv3[s, t] + m.v3[s, t] ** 2
    m.con1 = Constraint(m.s, m.t, rule=_diffeq)
    with self.assertRaises(DAE_Error):
        Simulator(m)
    m.del_component('con1')
    m.del_component('con1_index')