import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_05_empty_commit(self):
    """Commit of tree with no versioned files should fail"""
    self.make_branch_and_tree('.')
    self.build_tree(['hello.txt'])
    out, err = self.run_bzr('commit -m empty', retcode=3)
    self.assertEqual('', out)
    self.assertThat(err, DocTestMatches("Committing to: ...\nbrz: ERROR: No changes to commit. Please 'brz add' the files you want to commit, or use --unchanged to force an empty commit.\n", flags=doctest.ELLIPSIS | doctest.REPORT_UDIFF))