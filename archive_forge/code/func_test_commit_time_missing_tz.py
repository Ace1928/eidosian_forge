import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_time_missing_tz(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    out, err = self.run_bzr("commit -m hello --commit-time='2009-10-10 08:00:00' tree/hello.txt", retcode=3)
    self.assertStartsWith(err, 'brz: ERROR: Could not parse --commit-time:')
    self.assertContainsString(err, 'missing a timezone offset')