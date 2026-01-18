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
def test_bad_ext_ref_thin_pack(self):
    b1, b2, b3 = self.store_blobs([b'foo', b'bar', b'baz'])
    f = BytesIO()
    build_pack(f, [(REF_DELTA, (1, b'foo99')), (REF_DELTA, (b1.id, b'foo1')), (REF_DELTA, (b2.id, b'bar2')), (REF_DELTA, (b3.id, b'baz3'))], store=self.store)
    del self.store[b2.id]
    del self.store[b3.id]
    pack_iter = self.make_pack_iter(f)
    try:
        list(pack_iter._walk_all_chains())
        self.fail()
    except UnresolvedDeltas as e:
        self.assertEqual((sorted([b2.id, b3.id]),), (sorted(e.shas),))