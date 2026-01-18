import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_sanitizes_CR_in_message(self):
    a_tree = self.make_branch_and_tree('a')
    self.build_tree(['a/b'])
    a_tree.add('b')
    self.run_bzr(['commit', '-m', 'a string\r\n\r\nwith mixed\r\rendings\n'], working_dir='a')
    rev_id = a_tree.branch.last_revision()
    rev = a_tree.branch.repository.get_revision(rev_id)
    self.assertEqualDiff('a string\n\nwith mixed\n\nendings\n', rev.message)