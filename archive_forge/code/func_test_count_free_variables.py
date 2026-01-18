import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata
def test_count_free_variables(self):
    assert rpt.count_free_variables(self.m) == 7