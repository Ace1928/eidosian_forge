import logging
from itertools import product
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
def test_as_good_with_HCS_rule(self):
    """test that the high confidence stopping rule with very lenient
        parameters does no worse.
        """
    m = build_model()
    SolverFactory('ipopt').solve(m)
    for i in range(5):
        m2 = build_model()
        SolverFactory('multistart').solve(m2, iterations=-1, stopping_mass=0.99, stopping_delta=0.99)
        m_objectives = m.component_data_objects(Objective, active=True)
        m_obj = next(m_objectives, None)
        m2_objectives = m2.component_data_objects(Objective, active=True)
        m2_obj = next(m2_objectives, None)
        self.assertTrue(value(m2_obj.expr) >= value(m_obj.expr) - 0.001)
        del m2