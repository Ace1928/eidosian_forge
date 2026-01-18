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
def test_get_set_parent_ids(self):
    revision_tree, preview_tree = self.get_tree_and_preview_tree()
    self.assertEqual([], preview_tree.get_parent_ids())
    preview_tree.set_parent_ids([revision_tree.get_revision_id()])
    self.assertEqual([revision_tree.get_revision_id()], preview_tree.get_parent_ids())