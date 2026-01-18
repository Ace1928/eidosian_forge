import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
def test_add_blob_in_dir(self):
    blob_d = make_object(Blob, data=b'd')
    new_tree = commit_tree_changes(self.store, self.store[self.tree_id], [(b'e/f/d', 33188, blob_d.id)])
    self.assertEqual(new_tree.items(), [TreeEntry(path=b'a', mode=stat.S_IFREG | 33188, sha=self.blob_a.id), TreeEntry(path=b'ad', mode=stat.S_IFDIR, sha=b'0e2ce2cd7725ff4817791be31ccd6e627e801f4a'), TreeEntry(path=b'c', mode=stat.S_IFREG | 33188, sha=self.blob_c.id), TreeEntry(path=b'e', mode=stat.S_IFDIR, sha=b'6ab344e288724ac2fb38704728b8896e367ed108')])
    e_tree = self.store[new_tree[b'e'][1]]
    self.assertEqual(e_tree.items(), [TreeEntry(path=b'f', mode=stat.S_IFDIR, sha=b'24d2c94d8af232b15a0978c006bf61ef4479a0a5')])
    f_tree = self.store[e_tree[b'f'][1]]
    self.assertEqual(f_tree.items(), [TreeEntry(path=b'd', mode=stat.S_IFREG | 33188, sha=blob_d.id)])