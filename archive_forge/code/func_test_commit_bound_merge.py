import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_commit_bound_merge(self):
    master_branch = self.make_branch('master')
    bound_tree = self.make_branch_and_tree('bound')
    bound_tree.branch.bind(master_branch)
    self.build_tree_contents([('bound/content_file', b'initial contents\n')])
    bound_tree.add(['content_file'])
    bound_tree.commit(message='woo!')
    other_bzrdir = master_branch.controldir.sprout('other')
    other_tree = other_bzrdir.open_workingtree()
    self.build_tree_contents([('other/content_file', b'change in other\n')])
    other_tree.commit('change in other')
    bound_tree.merge_from_branch(other_tree.branch)
    self.build_tree_contents([('bound/content_file', b'change in bound\n')])
    bound_tree.commit(message='commit of merge in bound tree')