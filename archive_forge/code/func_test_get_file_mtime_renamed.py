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
def test_get_file_mtime_renamed(self):
    work_tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    work_tree.add('file')
    preview = work_tree.preview_transform()
    self.addCleanup(preview.finalize)
    file_trans_id = preview.trans_id_tree_path('file')
    preview.adjust_path('renamed', preview.root, file_trans_id)
    preview_tree = preview.get_preview_tree()
    preview_mtime = preview_tree.get_file_mtime('renamed')
    work_mtime = work_tree.get_file_mtime('file')