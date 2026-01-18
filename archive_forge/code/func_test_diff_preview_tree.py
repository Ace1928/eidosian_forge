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
def test_diff_preview_tree(self):
    revision_tree = self.create_tree()
    preview = revision_tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.new_file('file2', preview.root, [b'content B\n'], b'file2-id')
    preview_tree = preview.get_preview_tree()
    out = BytesIO()
    show_diff_trees(revision_tree, preview_tree, out)
    lines = out.getvalue().splitlines()
    self.assertEqual(lines[0], b"=== added file 'file2'")
    self.assertEqual(lines[4], b'+content B')