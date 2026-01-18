import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_empty_commit_message(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('foo.c', b'int main() {}')])
    tree.add('foo.c')
    self.run_bzr('commit -m ""')