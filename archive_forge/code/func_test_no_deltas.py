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
def test_no_deltas(self):
    f = BytesIO()
    entries = build_pack(f, [(Commit.type_num, b'commit'), (Blob.type_num, b'blob'), (Tree.type_num, b'tree')])
    self.assertEntriesMatch([0, 1, 2], entries, self.make_pack_iter(f))
    f.seek(0)
    self.assertEntriesMatch([], entries, self.make_pack_iter_subset(f, []))
    f.seek(0)
    self.assertEntriesMatch([1, 0], entries, self.make_pack_iter_subset(f, [entries[0][3], entries[1][3]]))
    f.seek(0)
    self.assertEntriesMatch([1, 0], entries, self.make_pack_iter_subset(f, [sha_to_hex(entries[0][3]), sha_to_hex(entries[1][3])]))