import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_file_size_too_large(self):
    with self.assertRaises(ValueError):
        minimum_part_size(40000 * 1024 * 1024 * 1024 + 1)