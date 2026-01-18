import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_kind_change(self):
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree_contents([('tree_a/file', b'content_1')])
    tree_a.add('file', ids=b'file-id')
    tree_a.commit('added file')
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    os.unlink('tree_a/file')
    self.build_tree(['tree_a/file/'])
    tree_a.commit('changed file to directory')
    self.run_bzr('merge ../tree_a', working_dir='tree_b')
    self.assertEqual('directory', osutils.file_kind('tree_b/file'))
    tree_b.revert()
    self.assertEqual('file', osutils.file_kind('tree_b/file'))
    self.build_tree_contents([('tree_b/file', b'content_2')])
    tree_b.commit('content change')
    self.run_bzr('merge ../tree_a', retcode=1, working_dir='tree_b')
    self.assertEqual(tree_b.conflicts(), [conflicts.ContentsConflict('file', file_id='file-id')])