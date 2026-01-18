import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_chunk_hashes_exact(self):
    chunks = chunk_hashes(b'a' * (2 * 1024 * 1024))
    self.assertEqual(len(chunks), 2)
    self.assertEqual(chunks[0], sha256(b'a' * 1024 * 1024).digest())