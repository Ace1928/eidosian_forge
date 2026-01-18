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
def evaluate_hessian(self, x):
    df0dx0dx0 = 2 - 4 * x[1] + 8 * x[1] ** 2
    df0dx0dx1 = -4 * x[0] - 2 * x[1] - 8 + 16 * x[0] * x[1] + 12 * x[1] ** 2
    df0dx1dx1 = -2 * x[0] - 8 + 8 * x[0] ** 2 + 24 * x[0] * x[1] + 12 * x[1] ** 2
    df1dx0dx0 = -2 * x[1]
    df1dx0dx1 = 2 * x[1] - 2 * x[0]
    df1dx1dx1 = 2 + 2 * x[0] + 6 * x[1]
    d2f0 = np.array([[df0dx0dx0, df0dx0dx1], [df0dx0dx1, df0dx1dx1]])
    d2f1 = np.array([[df1dx0dx0, df1dx0dx1], [df1dx0dx1, df1dx1dx1]])
    return [d2f0, d2f1]