import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_access_feature(self):
    feature = features.Feature()
    exception = tests.UnavailableFeature(feature)
    self.assertIs(feature, exception.args[0])