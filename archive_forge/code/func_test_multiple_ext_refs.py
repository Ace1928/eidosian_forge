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
def test_multiple_ext_refs(self):
    b1, b2 = self.store_blobs([b'foo', b'bar'])
    f = BytesIO()
    entries = build_pack(f, [(REF_DELTA, (b1.id, b'foo1')), (REF_DELTA, (b2.id, b'bar2'))], store=self.store)
    pack_iter = self.make_pack_iter(f)
    self.assertEntriesMatch([0, 1], entries, pack_iter)
    self.assertEqual([hex_to_sha(b1.id), hex_to_sha(b2.id)], pack_iter.ext_refs())