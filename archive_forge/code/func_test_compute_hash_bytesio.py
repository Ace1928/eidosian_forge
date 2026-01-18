import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_compute_hash_bytesio(self):
    f = BytesIO(self._gen_data())
    compute_hashes_from_fileobj(f, chunk_size=512)