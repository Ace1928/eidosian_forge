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
def make_pack_iter(self, f, thin=None):
    if thin is None:
        thin = bool(list(self.store))
    resolve_ext_ref = thin and self.get_raw_no_repeat or None
    data = PackData('test.pack', file=f)
    return TestPackIterator.for_pack_data(data, resolve_ext_ref=resolve_ext_ref)