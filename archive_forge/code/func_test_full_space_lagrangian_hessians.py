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
def test_full_space_lagrangian_hessians(self):
    model = Model2by2()
    m = model.make_model()
    m.x[0].set_value(1.0)
    m.x[1].set_value(2.0)
    m.y[0].set_value(3.0)
    m.y[1].set_value(4.0)
    x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
    x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
    lam_init_list = [-2.5, -0.5, 0.0, 1.0, 2.0]
    init_list = list(itertools.product(x0_init_list, x1_init_list, lam_init_list))
    external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
    for x0, x1, lam in init_list:
        x = [x0, x1]
        lam = [lam]
        external_model.set_input_values(x)
        external_model.set_external_constraint_multipliers(lam)
        hlxx, hlxy, hlyy = external_model.get_full_space_lagrangian_hessians()
        pred_hlxx, pred_hlxy, pred_hlyy = model.calculate_full_space_lagrangian_hessians(lam, x)
        np.testing.assert_allclose(hlxx.toarray(), pred_hlxx, rtol=1e-08)
        np.testing.assert_allclose(hlxy.toarray(), pred_hlxy, rtol=1e-08)
        np.testing.assert_allclose(hlyy.toarray(), pred_hlyy, rtol=1e-08)