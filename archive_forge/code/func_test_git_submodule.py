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
def test_git_submodule(self):
    repo_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, repo_dir)
    with Repo.init(repo_dir) as repo:
        filea = Blob.from_string(b'file alalala')
        subtree = Tree()
        subtree[b'a'] = (stat.S_IFREG | 420, filea.id)
        c = Commit()
        c.tree = subtree.id
        c.committer = c.author = b'Somebody <somebody@example.com>'
        c.commit_time = c.author_time = 42342
        c.commit_timezone = c.author_timezone = 0
        c.parents = []
        c.message = b'Subcommit'
        tree = Tree()
        tree[b'c'] = (S_IFGITLINK, c.id)
        repo.object_store.add_objects([(o, None) for o in [tree]])
        build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        index = repo.open_index()
        self.assertEqual(len(index), 1)
        apath = os.path.join(repo.path, 'c/a')
        self.assertFalse(os.path.exists(apath))
        cpath = os.path.join(repo.path, 'c')
        self.assertTrue(os.path.isdir(cpath))
        self.assertEqual(index[b'c'].mode, S_IFGITLINK)
        self.assertEqual(index[b'c'].sha, c.id)