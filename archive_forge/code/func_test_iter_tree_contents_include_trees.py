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
def test_iter_tree_contents_include_trees(self):
    blob_a = make_object(Blob, data=b'a')
    blob_b = make_object(Blob, data=b'b')
    blob_c = make_object(Blob, data=b'c')
    for blob in [blob_a, blob_b, blob_c]:
        self.store.add_object(blob)
    blobs = [(b'a', blob_a.id, 33188), (b'ad/b', blob_b.id, 33188), (b'ad/bd/c', blob_c.id, 33261)]
    tree_id = commit_tree(self.store, blobs)
    tree = self.store[tree_id]
    tree_ad = self.store[tree[b'ad'][1]]
    tree_bd = self.store[tree_ad[b'bd'][1]]
    expected = [TreeEntry(b'', 16384, tree_id), TreeEntry(b'a', 33188, blob_a.id), TreeEntry(b'ad', 16384, tree_ad.id), TreeEntry(b'ad/b', 33188, blob_b.id), TreeEntry(b'ad/bd', 16384, tree_bd.id), TreeEntry(b'ad/bd/c', 33261, blob_c.id)]
    actual = iter_tree_contents(self.store, tree_id, include_trees=True)
    self.assertEqual(expected, list(actual))