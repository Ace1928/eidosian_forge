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
def create_tree(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', b'content 1')])
    tree.add('a')
    revid1 = tree.commit('rev1')
    return tree.branch.repository.revision_tree(revid1)