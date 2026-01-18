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
def test_fs_to_tree_path_str(self):
    fs_path = os.path.join(os.path.join('délwíçh', 'foo'))
    tree_path = _fs_to_tree_path(fs_path)
    self.assertEqual(tree_path, 'délwíçh/foo'.encode())