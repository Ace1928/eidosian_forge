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
def test_iter_tree_contents(self):
    blob_a = make_object(Blob, data=b'a')
    blob_b = make_object(Blob, data=b'b')
    blob_c = make_object(Blob, data=b'c')
    for blob in [blob_a, blob_b, blob_c]:
        self.store.add_object(blob)
    blobs = [(b'a', blob_a.id, 33188), (b'ad/b', blob_b.id, 33188), (b'ad/bd/c', blob_c.id, 33261), (b'ad/c', blob_c.id, 33188), (b'c', blob_c.id, 33188)]
    tree_id = commit_tree(self.store, blobs)
    self.assertEqual([TreeEntry(p, m, h) for p, h, m in blobs], list(iter_tree_contents(self.store, tree_id)))
    self.assertEqual([], list(iter_tree_contents(self.store, None)))