import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def ruleMaker_1(m):
    return sum((m.c[i] * m.x[i] for i in m.I))