import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_available_module(self):
    feature = features.ModuleAvailableFeature('breezy.tests')
    self.assertEqual('breezy.tests', feature.module_name)
    self.assertEqual('breezy.tests', str(feature))
    self.assertTrue(feature.available())
    self.assertIs(tests, feature.module)