import os
import sys
from breezy import branch, osutils, registry, tests
def test_lazy_import_get_module(self):
    a_registry = registry.Registry()
    a_registry.register_lazy('obj', 'breezy.tests.test_registry', 'object1')
    self.assertEqual('breezy.tests.test_registry', a_registry._get_module('obj'))