import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_hook_template_rejected(self):
    tree = self.setup_commit_with_template()
    expected = tree.last_revision()
    out, err = self.run_bzr_error(['Empty commit message specified. Please specify a commit message with either --message or --file or leave a blank message with --message "".'], 'commit tree/hello.txt', stdin='n\n')
    self.assertEqual(expected, tree.last_revision())