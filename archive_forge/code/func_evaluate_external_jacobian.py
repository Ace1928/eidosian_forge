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
def evaluate_external_jacobian(self, x):
    dy0dx0 = -0.2 / (x[0] ** 2 * x[1])
    dy0dx1 = -0.2 / (x[0] * x[1] ** 2)
    dy1dx0 = -0.5 * x[1] ** 0.5 / x[0] ** 2
    dy1dx1 = 0.25 / (x[0] * x[1] ** 0.5)
    return np.array([[dy0dx0, dy0dx1], [dy1dx0, dy1dx1]])