import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_dynamic_backward_disc_with_initial_conditions(self):
    nfe = 5
    m = make_dynamic_model(nfe=nfe, scheme='BACKWARD')
    time = m.time
    t0 = m.time.first()
    t1 = m.time.next(t0)
    m.flow_in.fix()
    m.height[t0].fix()
    constraints = list(m.component_data_objects(pyo.Constraint))
    self.assertEqual(len(list(generate_strongly_connected_components(constraints))), nfe + 2)
    t_scc_map = {}
    for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints)):
        with TemporarySubsystemManager(to_fix=inputs):
            t = block.vars[0].index()
            t_scc_map[t] = i
            if t == t0:
                continue
            else:
                t_prev = m.time.prev(t)
                con_set = ComponentSet([m.diff_eqn[t], m.flow_out_eqn[t], m.dhdt_disc_eq[t]])
                var_set = ComponentSet([m.height[t], m.dhdt[t], m.flow_out[t]])
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(var_set), len(block.vars))
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                    self.assertFalse(var.fixed)
                other_var_set = ComponentSet([m.height[t_prev]]) if t != t1 else ComponentSet()
                self.assertEqual(len(inputs), len(other_var_set))
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)
                    self.assertTrue(var.fixed)
    scc = -1
    for t in m.time:
        if t == t0:
            self.assertTrue(m.height[t].fixed)
        else:
            self.assertFalse(m.height[t].fixed)
            self.assertGreater(t_scc_map[t], scc)
            scc = t_scc_map[t]
        self.assertFalse(m.flow_out[t].fixed)
        self.assertFalse(m.dhdt[t].fixed)
        self.assertTrue(m.flow_in[t].fixed)