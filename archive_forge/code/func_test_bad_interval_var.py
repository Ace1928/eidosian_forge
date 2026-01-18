import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_bad_interval_var(self):
    with self.assertRaisesRegex(TypeError, "The 'interval_var' argument for a 'Pulse' must be an 'IntervalVar'.\nReceived: <class 'float'>"):
        thing = Pulse(interval_var=1.2, height=4)