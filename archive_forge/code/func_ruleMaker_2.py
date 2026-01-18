import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def ruleMaker_2(m, i):
    if i >= 2:
        return m.x[i] <= 1
    else:
        return pyo.Constraint.Skip