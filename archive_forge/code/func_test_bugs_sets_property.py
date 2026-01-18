import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_bugs_sets_property(self):
    """commit --bugs=lp:234 sets the lp:234 revprop to 'related'."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr('commit -m hello --bugs=lp:234 tree/hello.txt')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    properties = dict(last_rev.properties)
    del properties['branch-nick']
    self.assertEqual({'bugs': 'https://launchpad.net/bugs/234 related'}, properties)