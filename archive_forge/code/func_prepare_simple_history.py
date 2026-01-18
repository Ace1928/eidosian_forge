import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def prepare_simple_history(self):
    """Prepare and return a working tree with one commit of one file"""
    wt = ControlDir.create_standalone_workingtree('.')
    self.build_tree(['hello.txt', 'extra.txt'])
    wt.add(['hello.txt'])
    wt.commit(message='added')
    return wt