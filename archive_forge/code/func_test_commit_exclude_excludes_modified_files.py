import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_exclude_excludes_modified_files(self):
    """Commit -x foo should ignore changes to foo."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b', 'c'])
    tree.smart_add(['.'])
    out, err = self.run_bzr(['commit', '-m', 'test', '-x', 'b'])
    self.assertFalse('added b' in out)
    self.assertFalse('added b' in err)
    out, err = self.run_bzr(['added'])
    self.assertEqual('b\n', out)
    self.assertEqual('', err)