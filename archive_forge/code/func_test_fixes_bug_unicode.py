import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_bug_unicode(self):
    """commit --fixes=lp:unicode succeeds without output."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    output, err = self.run_bzr_raw(['commit', '-m', 'hello', '--fixes=generic:€', 'tree/hello.txt'], encoding='utf-8', retcode=3)
    self.assertEqual(b'', output)
    self.assertContainsRe(err, b'brz: ERROR: Unrecognized bug generic:\xe2\x82\xac\\. Commit refused.\n')