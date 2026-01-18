import os
from breezy import ignores, osutils
from breezy.tests import TestCaseWithMemoryTransport
from breezy.tests.features import PluginLoadedFeature
def test_export_pot_plugin(self):
    self.requireFeature(PluginLoadedFeature('launchpad'))
    out, err = self.run_bzr('export-pot --plugin=launchpad')
    self.assertContainsRe(err, 'Exporting messages from plugin command: launchpad-login in launchpad')
    self.assertContainsRe(out, 'msgid "Show or set the Launchpad user ID."')