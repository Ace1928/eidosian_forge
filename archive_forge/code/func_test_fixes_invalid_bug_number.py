import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_invalid_bug_number(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr_error(['Did not understand bug identifier orange: Must be an integer. See "brz help bugs" for more information on this feature.\nCommit refused.'], 'commit -m add-b --fixes=lp:orange', working_dir='tree')