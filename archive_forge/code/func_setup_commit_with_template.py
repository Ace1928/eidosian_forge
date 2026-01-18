import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def setup_commit_with_template(self):
    self.setup_editor()
    msgeditor.hooks.install_named_hook('commit_message_template', lambda commit_obj, msg: 'save me some typing\n', None)
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    return tree