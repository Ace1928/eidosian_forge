import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_empty_tree_hash(self):
    self.assertEqual(self.calculate_tree_hash(''), b'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')