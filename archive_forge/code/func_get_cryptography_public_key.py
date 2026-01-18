from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from oslo_context import context
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.tests.unit.key_manager import mock_key_manager as mock_key_mgr
from castellan.tests.unit.key_manager import test_key_manager as test_key_mgr
def get_cryptography_public_key(public_key):
    crypto_public_key = serialization.load_der_public_key(bytes(public_key.get_encoded()), backend=backends.default_backend())
    return crypto_public_key