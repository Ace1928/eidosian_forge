from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from oslo_context import context
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.tests.unit.key_manager import mock_key_manager as mock_key_mgr
from castellan.tests.unit.key_manager import test_key_manager as test_key_mgr
def test_list_keys_metadata_only(self):
    key1 = sym_key.SymmetricKey('AES', 64 * 8, bytes(b'0' * 64))
    self.key_mgr.store(self.context, key1)
    key2 = sym_key.SymmetricKey('AES', 32 * 8, bytes(b'0' * 32))
    self.key_mgr.store(self.context, key2)
    keys = self.key_mgr.list(self.context, metadata_only=True)
    self.assertEqual(2, len(keys))
    bit_length_list = [key1.bit_length, key2.bit_length]
    for key in keys:
        self.assertTrue(key.is_metadata_only())
        self.assertIn(key.bit_length, bit_length_list)
    for key in keys:
        self.assertIsNotNone(key.id)