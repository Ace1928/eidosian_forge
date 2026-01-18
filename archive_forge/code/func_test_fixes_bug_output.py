import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_bug_output(self):
    """commit --fixes=lp:23452 succeeds without output."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    output, err = self.run_bzr('commit -m hello --fixes=lp:23452 tree/hello.txt')
    self.assertEqual('', output)
    self.assertContainsRe(err, 'Committing to: .*\nadded hello\\.txt\nCommitted revision 1\\.\n')