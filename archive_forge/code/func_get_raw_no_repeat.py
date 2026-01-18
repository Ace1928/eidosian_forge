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
def get_raw_no_repeat(self, bin_sha):
    """Wrapper around store.get_raw that doesn't allow repeat lookups."""
    hex_sha = sha_to_hex(bin_sha)
    self.assertNotIn(hex_sha, self.fetched, 'Attempted to re-fetch object %s' % hex_sha)
    self.fetched.add(hex_sha)
    return self.store.get_raw(hex_sha)