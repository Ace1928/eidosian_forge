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
def test_add_blob(self):
    blob_d = make_object(Blob, data=b'd')
    new_tree = commit_tree_changes(self.store, self.store[self.tree_id], [(b'd', 33188, blob_d.id)])
    self.assertEqual(new_tree[b'd'], (33188, b'c59d9b6344f1af00e504ba698129f07a34bbed8d'))