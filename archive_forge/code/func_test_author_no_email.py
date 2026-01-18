import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_author_no_email(self):
    """Author's name without an email address is allowed, too."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    out, err = self.run_bzr("commit -m hello --author='John Doe' tree/hello.txt")
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    properties = last_rev.properties
    self.assertEqual('John Doe', properties['authors'])