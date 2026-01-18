import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_fixes_invalid_argument(self):
    """Raise an appropriate error when the fixes argument isn't tag:id."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    self.run_bzr_error(['Invalid bug orange:apples:bananas. Must be in the form of \'tracker:id\'\\. See \\"brz help bugs\\" for more information on this feature.\\nCommit refused\\.'], 'commit -m add-b --fixes=orange:apples:bananas', working_dir='tree')