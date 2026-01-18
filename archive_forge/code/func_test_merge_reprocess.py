import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_reprocess(self):
    d = controldir.ControlDir.create_standalone_workingtree('.')
    d.commit('h')
    self.run_bzr('merge . --reprocess --merge-type weave')