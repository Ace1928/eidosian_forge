import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_gas_expansion(self):
    N = 5
    m = make_gas_expansion_model(N)
    m.rho[0].fix()
    m.F[0].fix()
    m.T[0].fix()
    constraints = list(m.component_data_objects(pyo.Constraint))
    self.assertEqual(len(list(generate_strongly_connected_components(constraints))), N + 1)
    for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints)):
        with TemporarySubsystemManager(to_fix=inputs):
            if i == 0:
                self.assertEqual(len(block.vars), 1)
                self.assertEqual(len(block.cons), 1)
                var_set = ComponentSet([m.P[i]])
                con_set = ComponentSet([m.ideal_gas[i]])
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                self.assertEqual(len(block.input_vars), 0)
            elif i == 1:
                self.assertEqual(len(block.vars), 4)
                self.assertEqual(len(block.cons), 4)
                var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                con_set = ComponentSet([m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]])
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                other_var_set = ComponentSet([m.P[i - 1]])
                self.assertEqual(len(block.input_vars), 1)
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)
            else:
                self.assertEqual(len(block.vars), 4)
                self.assertEqual(len(block.cons), 4)
                var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                con_set = ComponentSet([m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]])
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                other_var_set = ComponentSet([m.P[i - 1], m.rho[i - 1], m.F[i - 1], m.T[i - 1]])
                self.assertEqual(len(block.input_vars), 4)
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)