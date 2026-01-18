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
def test_ext_ref(self):
    blob, = self.store_blobs([b'blob'])
    f = BytesIO()
    entries = build_pack(f, [(REF_DELTA, (blob.id, b'blob1'))], store=self.store)
    pack_iter = self.make_pack_iter(f)
    self.assertEntriesMatch([0], entries, pack_iter)
    self.assertEqual([hex_to_sha(blob.id)], pack_iter.ext_refs())