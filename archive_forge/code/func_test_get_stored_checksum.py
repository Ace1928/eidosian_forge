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
def test_get_stored_checksum(self):
    p = self.get_pack_index(pack1_sha)
    self.assertEqual(b'f2848e2ad16f329ae1c92e3b95e91888daa5bd01', sha_to_hex(p.get_stored_checksum()))
    self.assertEqual(b'721980e866af9a5f93ad674144e1459b8ba3e7b7', sha_to_hex(p.get_pack_checksum()))