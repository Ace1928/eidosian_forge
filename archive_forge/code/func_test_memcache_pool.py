import uuid
import fixtures
from unittest import mock
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import utils
def test_memcache_pool(self):
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_use_advanced_pool': True}
    token = uuid.uuid4().hex.encode()
    data = uuid.uuid4().hex
    token_cache = self.create_simple_middleware(conf=conf)._token_cache
    token_cache.initialize({})
    token_cache.set(token, data)
    self.assertEqual(token_cache.get(token), data)