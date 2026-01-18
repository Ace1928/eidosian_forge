import os
import shutil
import sys
import tempfile
import zlib
from hashlib import sha1
from io import BytesIO
from typing import Set
from dulwich.tests import TestCase
from ..errors import ApplyDeltaError, ChecksumMismatch
from ..file import GitFile
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit, Tree, hex_to_sha, sha_to_hex
from ..pack import (
from .utils import build_pack, make_object
def test_iterobjects_2(self):
    with self.get_pack(pack1_sha) as p:
        objs = {o.id: o for o in p.iterobjects()}
        self.assertEqual(3, len(objs))
        self.assertEqual(sorted(objs), sorted(p.index))
        self.assertIsInstance(objs[a_sha], Blob)
        self.assertIsInstance(objs[tree_sha], Tree)
        self.assertIsInstance(objs[commit_sha], Commit)