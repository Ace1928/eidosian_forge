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
def test_decompress_empty(self):
    unpacked = UnpackedObject(Tree.type_num, decomp_len=0)
    comp = zlib.compress(b'')
    read = BytesIO(comp + self.extra).read
    unused = read_zlib_chunks(read, unpacked)
    self.assertEqual(b'', b''.join(unpacked.decomp_chunks))
    self.assertNotEqual(b'', unused)
    self.assertEqual(self.extra, unused + read())