import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_interactive_unlocks_branch(self):
    this = self.make_branch_and_tree('this')
    this.commit('empty commit')
    other = this.controldir.sprout('other').open_workingtree()
    other.commit('empty commit 2')
    self.run_bzr('merge -i -d this other')
    this.lock_write()
    this.unlock()