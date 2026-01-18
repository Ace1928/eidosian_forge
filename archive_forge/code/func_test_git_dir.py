import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
def test_git_dir(self):
    repo_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, repo_dir)
    with Repo.init(repo_dir) as repo:
        filea = Blob.from_string(b'file a')
        filee = Blob.from_string(b'd')
        tree = Tree()
        tree[b'.git/a'] = (stat.S_IFREG | 420, filea.id)
        tree[b'c/e'] = (stat.S_IFREG | 420, filee.id)
        repo.object_store.add_objects([(o, None) for o in [filea, filee, tree]])
        build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        index = repo.open_index()
        self.assertEqual(len(index), 1)
        apath = os.path.join(repo.path, '.git', 'a')
        self.assertFalse(os.path.exists(apath))
        epath = os.path.join(repo.path, 'c', 'e')
        self.assertTrue(os.path.exists(epath))
        self.assertReasonableIndexEntry(index[b'c/e'], stat.S_IFREG | 420, 1, filee.id)
        self.assertFileContents(epath, b'd')