import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_less_than_one_chunk(self):
    chunks = chunk_hashes(b'aaaa')
    self.assertEqual(len(chunks), 1)
    self.assertEqual(chunks[0], sha256(b'aaaa').digest())