import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_reversed_revision_range(self):
    self.run_bzr('merge -r 2..1 ' + self.context)
    self.assertPathDoesNotExist('a')
    self.assertPathExists('b')