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
def test_iterentries(self):
    with self.get_pack_data(pack1_sha) as p:
        entries = {(sha_to_hex(s), o, c) for s, o, c in p.iterentries()}
        self.assertEqual({(b'6f670c0fb53f9463760b7295fbb814e965fb20c8', 178, 1373561701), (b'b2a2766a2879c209ab1176e7e778b81ae422eeaa', 138, 912998690), (b'f18faa16531ac570a3fdc8c7ca16682548dafd12', 12, 3775879613)}, entries)