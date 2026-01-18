import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_default_format_option(self):
    """brz init should read default format from option default_format"""
    g_store = _mod_config.GlobalStore()
    g_store._load_from_string(b'\n[DEFAULT]\ndefault_format = 1.9\n')
    g_store.save()
    out, err = self.run_brz_subprocess('init')
    self.assertContainsRe(out, b'1.9')