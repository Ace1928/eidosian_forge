import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_lossy_native(self):
    """A --lossy option to commit is supported."""
    self.make_branch_and_tree('.')
    self.run_bzr('commit --lossy --unchanged -m message')
    self.assertEqual('', self.run_bzr('unknowns')[0])