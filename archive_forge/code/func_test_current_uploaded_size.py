from hashlib import sha256
import itertools
from boto.compat import StringIO
from tests.unit import unittest
from mock import (
from nose.tools import assert_equal
from boto.glacier.layer1 import Layer1
from boto.glacier.vault import Vault
from boto.glacier.writer import Writer, resume_file_upload
from boto.glacier.utils import bytes_to_hex, chunk_hashes, tree_hash
def test_current_uploaded_size(self):
    self.writer.write(b'1234')
    self.writer.write(b'567')
    size_1 = self.writer.current_uploaded_size
    self.assertEqual(size_1, 4)
    self.writer.write(b'22i3uy')
    size_2 = self.writer.current_uploaded_size
    self.assertEqual(size_2, 12)
    self.writer.close()
    final_size = self.writer.current_uploaded_size
    self.assertEqual(final_size, 13)
    self.assertEqual(final_size, self.writer.current_uploaded_size)