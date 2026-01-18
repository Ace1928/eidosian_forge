import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_lca_merge_criss_cross(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/file', b'base-contents\n')])
    tree_a.add('file')
    tree_a.commit('', rev_id=b'rev1')
    tree_b = tree_a.controldir.sprout('b').open_workingtree()
    self.build_tree_contents([('a/file', b'base-contents\nthis-contents\n')])
    tree_a.commit('', rev_id=b'rev2a')
    self.build_tree_contents([('b/file', b'base-contents\nother-contents\n')])
    tree_b.commit('', rev_id=b'rev2b')
    tree_a.merge_from_branch(tree_b.branch)
    self.build_tree_contents([('a/file', b'base-contents\nthis-contents\n')])
    tree_a.set_conflicts([])
    tree_b.merge_from_branch(tree_a.branch)
    self.build_tree_contents([('b/file', b'base-contents\nother-contents\n')])
    tree_b.set_conflicts([])
    tree_a.commit('', rev_id=b'rev3a')
    tree_b.commit('', rev_id=b'rev3b')
    out, err = self.run_bzr(['merge', '-d', 'a', 'b', '--lca'], retcode=1)
    self.assertFileEqual(b'base-contents\n<<<<<<< TREE\nthis-contents\n=======\nother-contents\n>>>>>>> MERGE-SOURCE\n', 'a/file')