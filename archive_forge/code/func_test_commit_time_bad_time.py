import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_time_bad_time(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    out, err = self.run_bzr("commit -m hello --commit-time='NOT A TIME' tree/hello.txt", retcode=3)
    self.assertStartsWith(err, 'brz: ERROR: Could not parse --commit-time:')