import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_with_uncommitted_changes(self):
    self.run_bzr_error(['Working tree .* has uncommitted changes'], ['merge', '../a'], working_dir='b')