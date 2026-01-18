import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_unknown_bug_prefix(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr_error(['Unrecognized bug %s. Commit refused.' % 'xxx:123'], 'commit -m add-b --fixes=xxx:123', working_dir='tree')