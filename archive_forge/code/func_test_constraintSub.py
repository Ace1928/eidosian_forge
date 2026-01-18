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
@unittest.skipIf(not opt.available(False), 'ipopt_sens is not available')
def test_constraintSub(self):
    m = ri.create_model()
    m.pert_a = Param(initialize=0.01)
    m.pert_b = Param(initialize=1.01)
    m_sipopt = sensitivity_calculation('sipopt', m, [m.a, m.b], [m.pert_a, m.pert_b])
    self.assertTrue(m_sipopt.C_equal.lower.ctype is Param and m_sipopt.C_equal.upper.ctype is Param)
    self.assertFalse(m_sipopt.C_equal.active)
    self.assertTrue(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[3].lower == 0.0 and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[3].upper == 0.0 and (len(list(identify_variables(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[3].body))) == 2))
    self.assertTrue(m_sipopt.C_singleBnd.lower is None and m_sipopt.C_singleBnd.upper.ctype is Param)
    self.assertFalse(m_sipopt.C_singleBnd.active)
    self.assertTrue(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[4].lower is None and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[4].upper == 0.0 and (len(list(identify_variables(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[4].body))) == 2))
    self.assertTrue(m_sipopt.C_rangedIn.lower.ctype is Param and m_sipopt.C_rangedIn.upper.ctype is Param)
    self.assertFalse(m_sipopt.C_rangedIn.active)
    self.assertTrue(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[1].lower is None and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[1].upper == 0.0 and (len(list(identify_variables(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[1].body))) == 2))
    self.assertTrue(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[2].lower is None and m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[2].upper == 0.0 and (len(list(identify_variables(m_sipopt._SENSITIVITY_TOOLBOX_DATA.constList[2].body))) == 2))