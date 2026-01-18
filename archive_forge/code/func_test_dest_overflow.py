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
def test_dest_overflow(self):
    self.assertRaises(ApplyDeltaError, apply_delta, b'a' * 65536, b'\x80\x80\x04\x80\x80\x04\x80' + b'a' * 65536)
    self.assertRaises(ApplyDeltaError, apply_delta, b'', b'\x00\x80\x02\xb0\x11\x11')