import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import math
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP, PyomoNLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
@m.Constraint(m.Tu)
def h2bal(m, t):
    return m.h2[t] - m.h2[t - 1] - dt / A2 * (c1 * pyo.sqrt(m.h1[t]) + m.F2[t] - c2 * pyo.sqrt(m.h2[t])) == 0