import os
from io import BytesIO
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_tree import TestCaseWithTree
from ... import revision as _mod_revision
from ... import tests, trace
from ...diff import show_diff_trees
from ...merge import Merge3Merger, Merger
from ...transform import ROOT_PARENT, resolve_conflicts
from ...tree import TreeChange, find_previous_path
from ..features import SymlinkFeature, UnicodeFilenameFeature
def test_merge_into_preview(self):
    work_tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file', b'b\n')])
    work_tree.add('file')
    work_tree.commit('first commit')
    child_tree = work_tree.controldir.sprout('child').open_workingtree()
    self.build_tree_contents([('child/file', b'b\nc\n')])
    child_tree.commit('child commit')
    child_tree.lock_write()
    self.addCleanup(child_tree.unlock)
    work_tree.lock_write()
    self.addCleanup(work_tree.unlock)
    preview = work_tree.preview_transform()
    self.addCleanup(preview.finalize)
    file_trans_id = preview.trans_id_tree_path('file')
    preview.delete_contents(file_trans_id)
    preview.create_file([b'a\nb\n'], file_trans_id)
    preview_tree = preview.get_preview_tree()
    merger = Merger.from_revision_ids(preview_tree, child_tree.branch.last_revision(), other_branch=child_tree.branch, tree_branch=work_tree.branch)
    merger.merge_type = Merge3Merger
    tt = merger.make_merger().make_preview_transform()
    self.addCleanup(tt.finalize)
    final_tree = tt.get_preview_tree()
    self.assertEqual(b'a\nb\nc\n', final_tree.get_file_text('file'))