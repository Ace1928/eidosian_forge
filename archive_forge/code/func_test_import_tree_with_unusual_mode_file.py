import os
import stat
import time
from dulwich.objects import S_IFGITLINK, Blob, Tag, Tree
from dulwich.repo import Repo as GitRepo
from ... import osutils
from ...branch import Branch
from ...bzr import knit, versionedfile
from ...bzr.inventory import Inventory
from ...controldir import ControlDir
from ...repository import Repository
from ...tests import TestCaseWithTransport
from ..fetch import import_git_blob, import_git_submodule, import_git_tree
from ..mapping import DEFAULT_FILE_MODE, BzrGitMappingv1
from . import GitBranchBuilder
def test_import_tree_with_unusual_mode_file(self):
    blob = Blob.from_string(b'bar1')
    tree = Tree()
    tree.add(b'foo', stat.S_IFREG | 436, blob.id)
    objects = {blob.id: blob, tree.id: tree}
    ret, child_modes = import_git_tree(self._texts, self._mapping, b'bla', b'bla', (None, tree.id), None, None, b'somerevid', [], objects.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
    self.assertEqual(child_modes, {b'bla/foo': stat.S_IFREG | 436})