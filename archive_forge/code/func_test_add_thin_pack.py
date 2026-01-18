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
def test_add_thin_pack(self):
    o = DiskObjectStore(self.store_dir)
    try:
        blob = make_object(Blob, data=b'yummy data')
        o.add_object(blob)
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (blob.id, b'more yummy data'))], store=o)
        with o.add_thin_pack(f.read, None) as pack:
            packed_blob_sha = sha_to_hex(entries[0][3])
            pack.check_length_and_checksum()
            self.assertEqual(sorted([blob.id, packed_blob_sha]), list(pack))
            self.assertTrue(o.contains_packed(packed_blob_sha))
            self.assertTrue(o.contains_packed(blob.id))
            self.assertEqual((Blob.type_num, b'more yummy data'), o.get_raw(packed_blob_sha))
    finally:
        o.close()