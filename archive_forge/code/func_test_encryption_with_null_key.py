import fixtures
import hashlib
import uuid
from oslo_log import log
from keystone.common import fernet_utils
from keystone.credential.providers import fernet as credential_fernet
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def test_encryption_with_null_key(self):
    null_key = fernet_utils.NULL_KEY
    null_key_hash = hashlib.sha1(null_key).hexdigest()
    blob = uuid.uuid4().hex
    encrypted_blob, primary_key_hash = self.provider.encrypt(blob)
    self.assertEqual(null_key_hash, primary_key_hash)
    self.assertNotEqual(blob, encrypted_blob)
    decrypted_blob = self.provider.decrypt(encrypted_blob)
    self.assertEqual(blob, decrypted_blob)