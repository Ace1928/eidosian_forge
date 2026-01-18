import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
def evaluate_residual(self, x):
    f0 = x[0] ** 2 + 2 * x[0] - 2 * x[0] ** 2 * x[1] - x[1] ** 2 * x[0] + 4 - 8 * x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[0] ** 2 * x[1] ** 2 + 4 * x[0] * x[1] ** 3 + x[1] ** 4 - 1.0
    f1 = x[1] ** 2 - x[1] + x[0] * x[1] ** 2 + x[1] ** 3 - x[0] ** 2 * x[1] - 2.0
    return (f0, f1)