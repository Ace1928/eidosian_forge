import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_recursive_checkout(self):
    """Ensure that a commit to a recursive checkout fails cleanly.
        """
    self.run_bzr(['init', 'test_branch'])
    self.run_bzr(['checkout', 'test_branch', 'test_checkout'])
    self.run_bzr(['bind', '.'], working_dir='test_checkout')
    with open('test_checkout/foo.txt', 'w') as f:
        f.write('hello')
    self.run_bzr(['add'], working_dir='test_checkout')
    out, err = self.run_bzr_error(['Branch.*test_checkout.*appears to be bound to itself'], ['commit', '-m', 'addedfoo'], working_dir='test_checkout')