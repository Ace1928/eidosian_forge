from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Param, Var, Block, Suffix, value
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap
from pyomo.core.expr import identify_variables
from pyomo.contrib.sensitivity_toolbox.sens import sipopt, kaug, sensitivity_calculation
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_ex
import pyomo.contrib.sensitivity_toolbox.examples.parameter_kaug as param_kaug_ex
import pyomo.contrib.sensitivity_toolbox.examples.feedbackController as fc
import pyomo.contrib.sensitivity_toolbox.examples.rangeInequality as ri
import pyomo.contrib.sensitivity_toolbox.examples.HIV_Transmission as hiv
@unittest.skipIf(not scipy_available, 'scipy is required for this test')
@unittest.skipIf(not opt.available(False), 'ipopt_sens is not available')
def test_clonedModel_soln(self):
    m_orig = fc.create_model()
    fc.initialize_model(m_orig, 100)
    m_orig.perturbed_a = Param(initialize=-0.25)
    m_orig.perturbed_H = Param(initialize=0.55)
    m_sipopt = sensitivity_calculation('sipopt', m_orig, [m_orig.a, m_orig.H], [m_orig.perturbed_a, m_orig.perturbed_H], cloneModel=True)
    self.assertFalse(m_sipopt == m_orig)
    self.assertTrue(hasattr(m_sipopt, '_SENSITIVITY_TOOLBOX_DATA') and m_sipopt._SENSITIVITY_TOOLBOX_DATA.ctype is Block)
    self.assertFalse(hasattr(m_orig, '_SENSITIVITY_TOOLBOX_DATA'))
    self.assertFalse(hasattr(m_orig, 'b'))
    self.assertTrue(hasattr(m_sipopt._SENSITIVITY_TOOLBOX_DATA, 'a') and m_sipopt._SENSITIVITY_TOOLBOX_DATA.a.ctype is Var)
    self.assertTrue(hasattr(m_sipopt._SENSITIVITY_TOOLBOX_DATA, 'H') and m_sipopt._SENSITIVITY_TOOLBOX_DATA.H.ctype is Var)
    self.assertTrue(hasattr(m_sipopt, 'sens_state_0') and m_sipopt.sens_state_0.ctype is Suffix and (m_sipopt.sens_state_0[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H] == 2) and (m_sipopt.sens_state_0[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a] == 1))
    self.assertTrue(hasattr(m_sipopt, 'sens_state_1') and m_sipopt.sens_state_1.ctype is Suffix and (m_sipopt.sens_state_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H] == 2) and (m_sipopt.sens_state_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a] == 1))
    self.assertTrue(hasattr(m_sipopt, 'sens_state_value_1') and m_sipopt.sens_state_value_1.ctype is Suffix and (m_sipopt.sens_state_value_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.H] == 0.55) and (m_sipopt.sens_state_value_1[m_sipopt._SENSITIVITY_TOOLBOX_DATA.a] == -0.25))
    self.assertTrue(hasattr(m_sipopt, 'sens_init_constr') and m_sipopt.sens_init_constr.ctype is Suffix and (m_sipopt.sens_init_constr[m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1]] == 1) and (m_sipopt.sens_init_constr[m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[2]] == 2))
    self.assertTrue(hasattr(m_sipopt, 'sens_sol_state_1') and m_sipopt.sens_sol_state_1.ctype is Suffix)
    self.assertAlmostEqual(m_sipopt.sens_sol_state_1[m_sipopt.F[15]], -0.00102016765, 8)
    self.assertTrue(hasattr(m_sipopt, 'sens_sol_state_1_z_L') and m_sipopt.sens_sol_state_1_z_L.ctype is Suffix)
    self.assertAlmostEqual(m_sipopt.sens_sol_state_1_z_L[m_sipopt.u[15]], -2.181712e-09, 13)
    self.assertTrue(hasattr(m_sipopt, 'sens_sol_state_1_z_U') and m_sipopt.sens_sol_state_1_z_U.ctype is Suffix)
    self.assertAlmostEqual(m_sipopt.sens_sol_state_1_z_U[m_sipopt.u[15]], 6.580899e-09, 13)
    self.assertFalse(m_sipopt.FDiffCon[0].active and m_sipopt.FDiffCon[7.5].active and m_sipopt.FDiffCon[15].active)
    self.assertFalse(m_sipopt.x_dot[0].active and m_sipopt.x_dot[7.5].active and m_sipopt.x_dot[15].active)
    self.assertTrue(m_orig.FDiffCon[0].active and m_orig.FDiffCon[7.5].active and m_orig.FDiffCon[15].active)
    self.assertTrue(m_orig.x_dot[0].active and m_orig.x_dot[7.5].active and m_orig.x_dot[15].active)
    self.assertAlmostEqual(value(m_sipopt.J), 0.0048956783, 8)