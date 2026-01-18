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
def check_write(self, write_list):
    for write_data in write_list:
        self.writer.write(write_data)
    self.writer.close()
    data = b''.join(write_list)
    upload_part_calls, data_tree_hashes = calculate_mock_vault_calls(data, self.part_size, self.chunk_size)
    check_mock_vault_calls(self.vault, upload_part_calls, data_tree_hashes, len(data))