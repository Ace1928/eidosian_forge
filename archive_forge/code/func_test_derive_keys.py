import struct
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.tests.unit import utils
def test_derive_keys(self):
    keys = self._setup_keys(b'strategy')
    self.assertEqual(len(keys['ENCRYPTION']), len(keys['CACHE_KEY']))
    self.assertEqual(len(keys['CACHE_KEY']), len(keys['MAC']))
    self.assertNotEqual(keys['ENCRYPTION'], keys['MAC'])
    self.assertIn('strategy', keys.keys())