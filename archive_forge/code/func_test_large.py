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
def test_large(self):
    entry1_sha = hex_to_sha('4e6388232ec39792661e2e75db8fb117fc869ce6')
    entry2_sha = hex_to_sha('e98f071751bd77f59967bfa671cd2caebdccc9a2')
    entries = [(entry1_sha, 17480489991855577991, 24), (entry2_sha, ~17480489991855577991 & 2 ** 64 - 1, 92)]
    if not self._supports_large:
        self.assertRaises(TypeError, self.index, 'single.idx', entries, pack_checksum)
        return
    idx = self.index('single.idx', entries, pack_checksum)
    self.assertEqual(idx.get_pack_checksum(), pack_checksum)
    self.assertEqual(2, len(idx))
    actual_entries = list(idx.iterentries())
    self.assertEqual(len(entries), len(actual_entries))
    for mine, actual in zip(entries, actual_entries):
        my_sha, my_offset, my_crc = mine
        actual_sha, actual_offset, actual_crc = actual
        self.assertEqual(my_sha, actual_sha)
        self.assertEqual(my_offset, actual_offset)
        if self._has_crc32_checksum:
            self.assertEqual(my_crc, actual_crc)
        else:
            self.assertIsNone(actual_crc)