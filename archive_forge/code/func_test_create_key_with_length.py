from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from oslo_context import context
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.tests.unit.key_manager import mock_key_manager as mock_key_mgr
from castellan.tests.unit.key_manager import test_key_manager as test_key_mgr
def test_create_key_with_length(self):
    for length in [64, 128, 256]:
        key_id = self.key_mgr.create_key(self.context, length=length)
        key = self.key_mgr.get(self.context, key_id)
        self.assertEqual(length / 8, len(key.get_encoded()))
        self.assertIsNotNone(key.id)