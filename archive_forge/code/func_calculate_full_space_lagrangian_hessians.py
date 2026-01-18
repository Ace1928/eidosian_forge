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
def calculate_full_space_lagrangian_hessians(self, lam, x):
    y = self.evaluate_external_variables(x)
    lam_g = self.calculate_external_multipliers(lam, x)
    d2fdx0dx0 = 2.0
    d2fdx1dx1 = 2.0
    d2fdy0dy0 = 2.0
    d2fdy1dy1 = 2.0
    hfxx = np.array([[d2fdx0dx0, 0], [0, d2fdx1dx1]])
    hfxy = np.array([[0, 0], [0, 0]])
    hfyy = np.array([[d2fdy0dy0, 0], [0, d2fdy1dy1]])
    dg0dx0dx0 = 2 * y[0] * x[1] ** 0.5 * y[1]
    dg0dx0dx1 = x[0] * y[0] * y[1] / x[1] ** 0.5
    dg0dx1dx1 = -1 / 4 * x[0] ** 2 * y[0] * y[1] / x[1] ** (3 / 2)
    dg0dx0dy0 = 2 * x[0] * x[1] ** 0.5 * y[1]
    dg0dx0dy1 = 2 * x[0] * y[0] * x[1] ** 0.5
    dg0dx1dy0 = 0.5 * x[0] ** 2 * y[1] / x[1] ** 0.5
    dg0dx1dy1 = 0.5 * x[0] ** 2 * y[0] / x[1] ** 0.5
    dg0dy0dy1 = x[0] ** 2 * x[1] ** 0.5
    hg0xx = np.array([[dg0dx0dx0, dg0dx0dx1], [dg0dx0dx1, dg0dx1dx1]])
    hg0xy = np.array([[dg0dx0dy0, dg0dx0dy1], [dg0dx1dy0, dg0dx1dy1]])
    hg0yy = np.array([[0, dg0dy0dy1], [dg0dy0dy1, 0]])
    dg1dx0dx1 = y[0]
    dg1dx0dy0 = x[1]
    dg1dx1dy0 = x[0]
    hg1xx = np.array([[0, dg1dx0dx1], [dg1dx0dx1, 0]])
    hg1xy = np.array([[dg1dx0dy0, 0], [dg1dx1dy0, 0]])
    hg1yy = np.zeros((2, 2))
    hlxx = lam[0] * hfxx + lam_g[0] * hg0xx + lam_g[1] * hg1xx
    hlxy = lam[0] * hfxy + lam_g[0] * hg0xy + lam_g[1] * hg1xy
    hlyy = lam[0] * hfyy + lam_g[0] * hg0yy + lam_g[1] * hg1yy
    return (hlxx, hlxy, hlyy)