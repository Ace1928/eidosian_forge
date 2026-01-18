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
def test_merge_preview_into_workingtree(self):
    tree = self.make_branch_and_tree('tree')
    if tree.supports_setting_file_ids():
        tree.set_root_id(b'TREE_ROOT')
    tt = tree.preview_transform()
    self.addCleanup(tt.finalize)
    tt.new_file('name', tt.root, [b'content'], b'file-id')
    tree2 = self.make_branch_and_tree('tree2')
    if tree.supports_setting_file_ids():
        tree2.set_root_id(b'TREE_ROOT')
    merger = Merger.from_uncommitted(tree2, tt.get_preview_tree(), tree.basis_tree())
    merger.merge_type = Merge3Merger
    merger.do_merge()