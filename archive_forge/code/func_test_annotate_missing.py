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
def test_annotate_missing(self):
    preview = self.get_empty_preview()
    preview.new_file('file', preview.root, [b'a\nb\nc\n'], b'file-id')
    preview_tree = preview.get_preview_tree()
    expected = [(b'me:', b'a\n'), (b'me:', b'b\n'), (b'me:', b'c\n')]
    annotation = preview_tree.annotate_iter('file', default_revision=b'me:')
    self.assertEqual(expected, annotation)