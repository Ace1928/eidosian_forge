from oslo_config import cfg
from heat.common import config
from heat.common import crypt
from heat.common import exception
from heat.tests import common
def test_fernet_key(self):
    key = 'x' * 16
    method, result = crypt.encrypt('foo', key)
    self.assertEqual('cryptography_decrypt_v1', method)
    self.assertIsNotNone(result)