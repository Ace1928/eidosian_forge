from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
def test_grouped_boxplot(self):
    graphics.grouped_boxplot(self.A, self.B, normalize=True, group_names=['A', 'B'])