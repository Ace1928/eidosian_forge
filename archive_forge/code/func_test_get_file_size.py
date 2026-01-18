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
def test_get_file_size(self):
    work_tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/old', b'old')])
    work_tree.add('old')
    preview = work_tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.new_file('name', preview.root, [b'contents'], b'new-id', 'executable')
    tree = preview.get_preview_tree()
    self.assertEqual(len('old'), tree.get_file_size('old'))
    self.assertEqual(len('contents'), tree.get_file_size('name'))