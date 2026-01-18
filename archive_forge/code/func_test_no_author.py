import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_no_author(self):
    """If the author is not specified, the author property is not set."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr('commit -m hello tree/hello.txt')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    properties = last_rev.properties
    self.assertFalse('author' in properties)