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
def test_check_getitemexpression(self):
    m = self.m
    t = IndexTemplate(m.t)
    e = m.dv[t] == m.v[t]
    temp = _check_getitemexpression(e, 0)
    self.assertIs(e.arg(0), temp[0])
    self.assertIs(e.arg(1), temp[1])
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(m.v, temp[1].arg(0))
    temp = _check_getitemexpression(e, 1)
    self.assertIsNone(temp)
    e = m.v[t] == m.dv[t]
    temp = _check_getitemexpression(e, 1)
    self.assertIs(e.arg(0), temp[1])
    self.assertIs(e.arg(1), temp[0])
    self.assertIs(m.dv, temp[0].arg(0))
    self.assertIs(m.v, temp[1].arg(0))
    temp = _check_getitemexpression(e, 0)
    self.assertIsNone(temp)
    e = m.v[t] == m.v[t]
    temp = _check_getitemexpression(e, 0)
    self.assertIsNone(temp)
    temp = _check_getitemexpression(e, 1)
    self.assertIsNone(temp)