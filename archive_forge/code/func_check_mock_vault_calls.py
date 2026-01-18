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
def check_mock_vault_calls(vault, upload_part_calls, data_tree_hashes, data_len):
    vault.layer1.upload_part.assert_has_calls(upload_part_calls, any_order=True)
    assert_equal(len(upload_part_calls), vault.layer1.upload_part.call_count)
    data_tree_hash = bytes_to_hex(tree_hash(data_tree_hashes))
    vault.layer1.complete_multipart_upload.assert_called_once_with(sentinel.vault_name, sentinel.upload_id, data_tree_hash, data_len)