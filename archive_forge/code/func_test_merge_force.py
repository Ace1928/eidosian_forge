import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_force(self):
    self.tree_a.commit('empty change to allow merge to run')
    self.run_bzr(['merge', '../a', '--force'], working_dir='b')