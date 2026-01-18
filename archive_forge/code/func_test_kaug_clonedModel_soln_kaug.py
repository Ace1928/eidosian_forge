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
@unittest.skipIf(not opt_kaug.available(False), 'k_aug is not available')
@unittest.skipIf(not opt_dotsens.available(False), 'dot_sens is not available')
def test_kaug_clonedModel_soln_kaug(self):
    m_orig = fc.create_model()
    fc.initialize_model(m_orig, 100)
    m_orig.perturbed_a = Param(initialize=-0.25)
    m_orig.perturbed_H = Param(initialize=0.55)
    m_kaug = sensitivity_calculation('k_aug', m_orig, [m_orig.a, m_orig.H], [m_orig.perturbed_a, m_orig.perturbed_H], cloneModel=True)
    ptb_map = ComponentMap()
    ptb_map[m_kaug.a] = value(-(m_orig.perturbed_a - m_orig.a))
    ptb_map[m_kaug.H] = value(-(m_orig.perturbed_H - m_orig.H))
    self.assertIsNot(m_kaug, m_orig)
    self.assertTrue(hasattr(m_kaug, '_SENSITIVITY_TOOLBOX_DATA') and m_kaug._SENSITIVITY_TOOLBOX_DATA.ctype is Block)
    self.assertFalse(hasattr(m_orig, '_SENSITIVITY_TOOLBOX_DATA'))
    self.assertFalse(hasattr(m_orig, 'b'))
    self.assertTrue(hasattr(m_kaug._SENSITIVITY_TOOLBOX_DATA, 'a') and m_kaug._SENSITIVITY_TOOLBOX_DATA.a.ctype is Var)
    self.assertTrue(hasattr(m_kaug._SENSITIVITY_TOOLBOX_DATA, 'H') and m_kaug._SENSITIVITY_TOOLBOX_DATA.H.ctype is Var)
    self.assertTrue(hasattr(m_kaug, 'sens_state_0') and m_kaug.sens_state_0.ctype is Suffix and (m_kaug.sens_state_0[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 2) and (m_kaug.sens_state_0[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == 1))
    self.assertTrue(hasattr(m_kaug, 'sens_state_1') and m_kaug.sens_state_1.ctype is Suffix and (m_kaug.sens_state_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 2) and (m_kaug.sens_state_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == 1))
    self.assertTrue(hasattr(m_kaug, 'sens_state_value_1') and m_kaug.sens_state_value_1.ctype is Suffix and (m_kaug.sens_state_value_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.H] == 0.55) and (m_kaug.sens_state_value_1[m_kaug._SENSITIVITY_TOOLBOX_DATA.a] == -0.25))
    self.assertTrue(hasattr(m_kaug, 'sens_init_constr') and m_kaug.sens_init_constr.ctype is Suffix and (m_kaug.sens_init_constr[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]] == 1) and (m_kaug.sens_init_constr[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]] == 2))
    self.assertTrue(hasattr(m_kaug, 'DeltaP'))
    self.assertTrue(m_kaug.DeltaP.ctype is Suffix)
    self.assertEqual(m_kaug.DeltaP[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]], ptb_map[m_kaug.a])
    self.assertEqual(m_kaug.DeltaP[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]], ptb_map[m_kaug.H])
    self.assertTrue(hasattr(m_kaug, 'dcdp') and m_kaug.dcdp.ctype is Suffix and (m_kaug.dcdp[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[1]] == 1) and (m_kaug.dcdp[m_kaug._SENSITIVITY_TOOLBOX_DATA.paramConst[2]] == 2))
    self.assertTrue(hasattr(m_kaug, 'sens_sol_state_1') and m_kaug.sens_sol_state_1.ctype is Suffix)
    self.assertTrue(hasattr(m_kaug, 'ipopt_zL_in') and m_kaug.ipopt_zL_in.ctype is Suffix)
    self.assertAlmostEqual(m_kaug.ipopt_zL_in[m_kaug.u[15]], 7.162686166847096e-09, 13)
    self.assertTrue(hasattr(m_kaug, 'ipopt_zU_in') and m_kaug.ipopt_zU_in.ctype is Suffix)
    self.assertAlmostEqual(m_kaug.ipopt_zU_in[m_kaug.u[15]], -1.2439730261288605e-08, 13)
    self.assertFalse(m_kaug.FDiffCon[0].active and m_kaug.FDiffCon[7.5].active and m_kaug.FDiffCon[15].active)
    self.assertFalse(m_kaug.x_dot[0].active and m_kaug.x_dot[7.5].active and m_kaug.x_dot[15].active)
    self.assertTrue(m_orig.FDiffCon[0].active and m_orig.FDiffCon[7.5].active and m_orig.FDiffCon[15].active)
    self.assertTrue(m_orig.x_dot[0].active and m_orig.x_dot[7.5].active and m_orig.x_dot[15].active)
    self.assertAlmostEqual(value(m_kaug.J), 0.002633263921107476, 8)