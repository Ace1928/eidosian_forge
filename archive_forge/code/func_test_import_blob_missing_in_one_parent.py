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
def test_import_blob_missing_in_one_parent(self):
    builder = self.make_branch_builder('br')
    builder.start_series()
    rev_root = builder.build_snapshot(None, [('add', ('', b'rootid', 'directory', ''))])
    rev1 = builder.build_snapshot([rev_root], [('add', ('bla', self._mapping.generate_file_id('bla'), 'file', b'content'))])
    rev2 = builder.build_snapshot([rev_root], [])
    builder.finish_series()
    branch = builder.get_branch()
    blob = Blob.from_string(b'bar')
    objs = {'blobname': blob}
    ret = import_git_blob(self._texts, self._mapping, b'bla', b'bla', (None, 'blobname'), branch.repository.revision_tree(rev1), b'rootid', b'somerevid', [branch.repository.revision_tree(r) for r in [rev1, rev2]], objs.__getitem__, (None, DEFAULT_FILE_MODE), DummyStoreUpdater(), self._mapping.generate_file_id)
    self.assertEqual({(b'git:bla', b'somerevid')}, self._texts.keys())