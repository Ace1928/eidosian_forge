import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_does_not_store(self):
    self.prepare()
    self.run_bzr(['switch', '-d', 'checkout', 'new'])
    self.assertPathExists('checkout/a')