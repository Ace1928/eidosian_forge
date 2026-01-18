import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_create_prefix(self):
    """'brz init --create-prefix; will create leading directories."""
    tree = self.create_simple_tree()
    self.run_bzr_error(['Parent directory of ../new/tree does not exist'], 'init ../new/tree', working_dir='tree')
    self.run_bzr('init ../new/tree --create-prefix', working_dir='tree')
    self.assertPathExists('new/tree/.bzr')