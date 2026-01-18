import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_bug_with_default_tracker(self):
    """commit --fixes=234 uses the default bug tracker."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr_error(['brz: ERROR: No tracker specified for bug 123. Use the form \'tracker:id\' or specify a default bug tracker using the `bugtracker` option.\nSee "brz help bugs" for more information on this feature. Commit refused.'], 'commit -m add-b --fixes=123', working_dir='tree')
    tree.branch.get_config_stack().set('bugtracker', 'lp')
    self.run_bzr('commit -m hello --fixes=234 tree/hello.txt')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    self.assertEqual('https://launchpad.net/bugs/234 fixed', last_rev.properties['bugs'])