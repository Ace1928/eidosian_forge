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
def test_iter_unpacked(self):
    with self.get_pack_data(pack1_sha) as p:
        commit_data = b'tree b2a2766a2879c209ab1176e7e778b81ae422eeaa\nauthor James Westby <jw+debian@jameswestby.net> 1174945067 +0100\ncommitter James Westby <jw+debian@jameswestby.net> 1174945067 +0100\n\nTest commit\n'
        blob_sha = b'6f670c0fb53f9463760b7295fbb814e965fb20c8'
        tree_data = b'100644 a\x00' + hex_to_sha(blob_sha)
        actual = list(p.iter_unpacked())
        self.assertEqual([UnpackedObject(offset=12, pack_type_num=1, decomp_chunks=[commit_data], crc32=None), UnpackedObject(offset=138, pack_type_num=2, decomp_chunks=[tree_data], crc32=None), UnpackedObject(offset=178, pack_type_num=3, decomp_chunks=[b'test 1\n'], crc32=None)], actual)