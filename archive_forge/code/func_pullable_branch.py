import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def pullable_branch(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/file', b'bar\n')])
    tree_a.add(['file'])
    self.id1 = tree_a.commit('commit 1')
    tree_b = self.make_branch_and_tree('b')
    tree_b.pull(tree_a.branch)
    self.build_tree_contents([('b/file', b'foo\n')])
    self.id2 = tree_b.commit('commit 2')