import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_cannot_add_to_constant(self):
    m = self.get_model()
    with self.assertRaisesRegex(TypeError, "Cannot add object of class <class 'pyomo.contrib.cp.scheduling_expr.step_function_expressions.StepAtStart'> to object of class <class 'int'>"):
        expr = 4 + Step(m.a.start_time, height=6)