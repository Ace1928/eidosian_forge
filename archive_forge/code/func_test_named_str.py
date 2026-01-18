import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_named_str(self):
    """Feature.__str__ should thunk to feature_name()."""

    class NamedFeature(features.Feature):

        def feature_name(self):
            return 'symlinks'
    feature = NamedFeature()
    self.assertEqual('symlinks', str(feature))