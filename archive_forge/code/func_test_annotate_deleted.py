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
def test_annotate_deleted(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file', b'a\n')])
    tree.add('file')
    tree.commit('a')
    self.build_tree_contents([('tree/file', b'a\nb\n')])
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    file_trans_id = preview.trans_id_tree_path('file')
    preview.delete_contents(file_trans_id)
    preview_tree = preview.get_preview_tree()
    annotation = preview_tree.annotate_iter('file', default_revision=b'me:')
    self.assertIs(None, annotation)