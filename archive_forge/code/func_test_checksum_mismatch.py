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
def test_checksum_mismatch(self):
    with self.get_pack_data(pack1_sha) as data:
        index = self.get_pack_index(pack1_sha)
        Pack.from_objects(data, index).check_length_and_checksum()
        data._file.seek(0)
        bad_file = BytesIO(data._file.read()[:-20] + b'\xff' * 20)
        bad_data = PackData('', file=bad_file)
        bad_pack = Pack.from_lazy_objects(lambda: bad_data, lambda: index)
        self.assertRaises(ChecksumMismatch, lambda: bad_pack.data)
        self.assertRaises(ChecksumMismatch, bad_pack.check_length_and_checksum)