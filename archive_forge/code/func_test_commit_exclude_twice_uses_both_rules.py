import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_exclude_twice_uses_both_rules(self):
    """Commit -x foo -x bar should ignore changes to foo and bar."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b', 'c'])
    tree.smart_add(['.'])
    out, err = self.run_bzr(['commit', '-m', 'test', '-x', 'b', '-x', 'c'])
    self.assertFalse('added b' in out)
    self.assertFalse('added c' in out)
    self.assertFalse('added b' in err)
    self.assertFalse('added c' in err)
    out, err = self.run_bzr(['added'])
    self.assertTrue('b\n' in out)
    self.assertTrue('c\n' in out)
    self.assertEqual('', err)