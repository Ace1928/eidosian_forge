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
def test_compute_file_sha(self):
    f = BytesIO(b'abcd1234wxyz')
    self.assertEqual(sha1(b'abcd1234wxyz').hexdigest(), compute_file_sha(f).hexdigest())
    self.assertEqual(sha1(b'abcd1234wxyz').hexdigest(), compute_file_sha(f, buffer_size=5).hexdigest())
    self.assertEqual(sha1(b'abcd1234').hexdigest(), compute_file_sha(f, end_ofs=-4).hexdigest())
    self.assertEqual(sha1(b'1234wxyz').hexdigest(), compute_file_sha(f, start_ofs=4).hexdigest())
    self.assertEqual(sha1(b'1234').hexdigest(), compute_file_sha(f, start_ofs=4, end_ofs=-4).hexdigest())