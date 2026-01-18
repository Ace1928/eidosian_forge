import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
def test_generate_subsystems_dont_fix_inputs(self):
    m = _make_simple_model()
    subs = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
    other_vars = [[m.v2, m.v3, m.v4], [m.v1, m.v4]]
    for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
        self.assertIs(block.model(), block)
        var_set = ComponentSet(subs[i][1])
        con_set = ComponentSet(subs[i][0])
        input_set = ComponentSet(other_vars[i])
        self.assertEqual(len(var_set), len(block.vars))
        self.assertEqual(len(con_set), len(block.cons))
        self.assertEqual(len(input_set), len(inputs))
        self.assertTrue(all((var in var_set for var in block.vars[:])))
        self.assertTrue(all((con in con_set for con in block.cons[:])))
        self.assertTrue(all((var in input_set for var in inputs)))
        self.assertFalse(any((var.fixed for var in inputs)))
        self.assertFalse(any((var.fixed for var in block.vars[:])))
    self.assertFalse(m.v1.fixed)
    self.assertFalse(m.v2.fixed)
    self.assertFalse(m.v3.fixed)
    self.assertFalse(m.v4.fixed)