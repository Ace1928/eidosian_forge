import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_accelerator_tree_observes_sha1(self):
    source = self.create_ab_tree()
    sha1 = osutils.sha_string(b'A')
    target = self.make_branch_and_tree('target')
    target.lock_write()
    self.addCleanup(target.unlock)
    state = target.current_dirstate()
    state._cutoff_time = time.time() + 60
    build_tree(source.basis_tree(), target, source)
    entry = state._get_entry(0, path_utf8=b'file1')
    self.assertEqual(sha1, entry[1][0][1])