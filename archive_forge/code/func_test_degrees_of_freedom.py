import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata
def test_degrees_of_freedom(self):
    assert rpt.degrees_of_freedom(self.m) == 0