from pyomo.common import unittest
from pyomo.contrib.solver.config import (
def test_interface_custom_instantiation(self):
    config = PersistentSolverConfig(description='A description')
    config.tee = True
    config.auto_updates.check_for_new_objective = False
    self.assertTrue(config.tee)
    self.assertEqual(config._description, 'A description')
    self.assertFalse(config.auto_updates.check_for_new_objective)