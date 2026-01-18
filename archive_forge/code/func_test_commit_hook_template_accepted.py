import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_hook_template_accepted(self):
    tree = self.setup_commit_with_template()
    out, err = self.run_bzr('commit tree/hello.txt', stdin='y\n')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    self.assertEqual('save me some typing\n', last_rev.message)