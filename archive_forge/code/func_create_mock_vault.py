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
def create_mock_vault():
    vault = Mock(spec=Vault)
    vault.layer1 = Mock(spec=Layer1)
    vault.layer1.complete_multipart_upload.return_value = dict(ArchiveId=sentinel.archive_id)
    vault.name = sentinel.vault_name
    return vault