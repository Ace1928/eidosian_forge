import os
from breezy import ignores, osutils
from breezy.tests import TestCaseWithMemoryTransport
from breezy.tests.features import PluginLoadedFeature
def test_export_pot(self):
    out, err = self.run_bzr('export-pot')
    self.assertContainsRe(err, 'Exporting messages from builtin command: add')
    self.assertContainsRe(out, 'help of \'change\' option\nmsgid "Select changes introduced by the specified revision.')