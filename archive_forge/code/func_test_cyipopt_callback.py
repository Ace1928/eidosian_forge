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
@unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
def test_cyipopt_callback(self):
    m = self.make_model()
    scaling_factors = [0.0001, 10000.0]
    m.epm.set_equality_constraint_scaling_factors(scaling_factors)
    nlp = PyomoNLPWithGreyBoxBlocks(m)

    def callback(local_nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        primals = tuple(local_nlp.get_primals())
        u, v, x, y = primals
        con_3_resid = scaling_factors[0] * abs(self.con_3_body(x, y, u, v) - self.con_3_rhs())
        con_4_resid = scaling_factors[1] * abs(self.con_4_body(x, y, u, v) - self.con_4_rhs())
        pred_inf_pr = max(con_3_resid, con_4_resid)
        self.assertAlmostEqual(inf_pr, pred_inf_pr)
    cyipopt_nlp = CyIpoptNLP(nlp, intermediate_callback=callback)
    x0 = nlp.get_primals()
    cyipopt = CyIpoptSolver(cyipopt_nlp, options={'max_iter': 0, 'nlp_scaling_method': 'user-scaling'})
    cyipopt.solve(x0=x0)