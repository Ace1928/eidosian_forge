import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
def calculate_tree_hash(self, bytestring):
    start = time.time()
    calculated = bytes_to_hex(tree_hash(chunk_hashes(bytestring)))
    end = time.time()
    logging.debug('Tree hash calc time for length %s: %s', len(bytestring), end - start)
    return calculated