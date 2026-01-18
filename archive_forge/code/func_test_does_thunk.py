import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_does_thunk(self):
    res = self.callDeprecated(['breezy.tests.test_features.simple_thunk_feature was deprecated in version 2.1.0. Use breezy.tests.features.UnicodeFilenameFeature instead.'], simple_thunk_feature.available)
    self.assertEqual(features.UnicodeFilenameFeature.available(), res)