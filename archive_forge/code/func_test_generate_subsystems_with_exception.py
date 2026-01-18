import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
def test_generate_subsystems_with_exception(self):
    m = _make_simple_model()
    subsystems = [([m.con1], [m.v1, m.v4]), ([m.con2, m.con3], [m.v2, m.v3])]
    other_vars = [[m.v2, m.v3], [m.v1, m.v4]]
    block = create_subsystem_block(*subsystems[0])
    with self.assertRaises(RuntimeError):
        inputs = list(block.input_vars[:])
        with TemporarySubsystemManager(to_fix=inputs):
            self.assertTrue(all((var.fixed for var in inputs)))
            self.assertFalse(any((var.fixed for var in block.vars[:])))
            raise RuntimeError()
    self.assertFalse(any((var.fixed for var in m.component_data_objects(pyo.Var))))