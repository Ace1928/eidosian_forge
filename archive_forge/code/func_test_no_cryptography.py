import struct
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.tests.unit import utils
def test_no_cryptography(self):
    aes = memcache_crypt.ciphers
    memcache_crypt.ciphers = None
    self.assertRaises(memcache_crypt.CryptoUnavailableError, memcache_crypt.encrypt_data, 'token', 'secret', 'data')
    memcache_crypt.ciphers = aes