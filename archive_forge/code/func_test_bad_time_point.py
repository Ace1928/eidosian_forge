import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_bad_time_point(self):
    m = self.get_model()
    with self.assertRaisesRegex(TypeError, "The 'time' argument for a 'Step' must be either an 'IntervalVarTimePoint' \\(for example, the 'start_time' or 'end_time' of an IntervalVar\\) or an integer time point in the time horizon.\nReceived: <class 'pyomo.contrib.cp.interval_var.ScalarIntervalVar'>"):
        thing = Step(m.a, height=2)