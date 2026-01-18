import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata
def test_count_equality_constraints(self):
    assert rpt.count_equality_constraints(self.m) == 7