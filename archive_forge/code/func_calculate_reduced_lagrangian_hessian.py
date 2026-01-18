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
def calculate_reduced_lagrangian_hessian(self, lam, x):
    dydx = self.evaluate_external_jacobian(x)
    hlxx, hlxy, hlyy = self.calculate_full_space_lagrangian_hessians(lam, x)
    return hlxx + hlxy.dot(dydx).transpose() + hlxy.dot(dydx) + dydx.transpose().dot(hlyy).dot(dydx)