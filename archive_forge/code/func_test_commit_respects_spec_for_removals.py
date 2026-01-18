import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_commit_respects_spec_for_removals(self):
    """Commit with a file spec should only commit removals that match"""
    t = self.make_branch_and_tree('.')
    self.build_tree(['file-a', 'dir-a/', 'dir-a/file-b'])
    t.add(['file-a', 'dir-a', 'dir-a/file-b'])
    t.commit('Create')
    t.remove(['file-a', 'dir-a/file-b'])
    result = self.run_bzr('commit . -m removed-file-b', working_dir='dir-a')[1]
    self.assertNotContainsRe(result, 'file-a')
    result = self.run_bzr('status', working_dir='dir-a')[0]
    self.assertContainsRe(result, 'removed:\n  file-a')