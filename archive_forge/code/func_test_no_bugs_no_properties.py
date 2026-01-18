import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_no_bugs_no_properties(self):
    """If no bugs are fixed, the bugs property is not set.

        see https://beta.launchpad.net/bzr/+bug/109613
        """
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr('commit -m hello tree/hello.txt')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    properties = dict(last_rev.properties)
    del properties['branch-nick']
    self.assertFalse('bugs' in properties)