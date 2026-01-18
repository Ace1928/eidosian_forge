import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata
def test_active_equality_set(self):
    self.m.c4.deactivate()
    assert rpt.active_equality_set(self.m) == ComponentSet([self.m.c1, self.m.c2, self.m.c3, self.m.c5, self.m.c6, self.m.c7])
    self.m.c4.activate()
    assert rpt.active_equality_set(self.m) == ComponentSet([self.m.c1, self.m.c2, self.m.c3, self.m.c4, self.m.c5, self.m.c6, self.m.c7])