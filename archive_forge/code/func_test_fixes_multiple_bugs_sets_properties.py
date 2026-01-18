import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_multiple_bugs_sets_properties(self):
    """--fixes can be used more than once to show that bugs are fixed."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr('commit -m hello --fixes=lp:123 --fixes=lp:235 tree/hello.txt')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    properties = dict(last_rev.properties)
    del properties['branch-nick']
    self.assertEqual({'bugs': 'https://launchpad.net/bugs/123 fixed\nhttps://launchpad.net/bugs/235 fixed'}, properties)