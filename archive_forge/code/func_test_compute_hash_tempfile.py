import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def test_compute_hash_tempfile(self):
    if six.PY2:
        mode = 'w+'
    else:
        mode = 'wb+'
    with tempfile.TemporaryFile(mode=mode) as f:
        f.write(self._gen_data())
        f.seek(0)
        compute_hashes_from_fileobj(f, chunk_size=512)