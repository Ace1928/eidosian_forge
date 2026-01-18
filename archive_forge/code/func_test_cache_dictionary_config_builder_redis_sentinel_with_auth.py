import copy
import ssl
import time
from unittest import mock
from dogpile.cache import proxy
from oslo_config import cfg
from oslo_utils import uuidutils
from pymemcache import KeepaliveOpts
from oslo_cache import _opts
from oslo_cache import core as cache
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_cache_dictionary_config_builder_redis_sentinel_with_auth(self):
    """Validate the backend is reset to default if caching is disabled."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.redis_sentinel', redis_username='user', redis_password='secrete', redis_sentinels=['127.0.0.1:26379', '[::1]:26379', 'localhost:26379'], redis_sentinel_service_name='cluster')
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertFalse(self.config_fixture.conf.cache.tls_enabled)
    self.assertEqual('cluster', config_dict['test_prefix.arguments.service_name'])
    self.assertEqual([('127.0.0.1', 26379), ('::1', 26379), ('localhost', 26379)], config_dict['test_prefix.arguments.sentinels'])
    self.assertEqual('secrete', config_dict['test_prefix.arguments.password'])
    self.assertEqual({'username': 'user'}, config_dict['test_prefix.arguments.connection_kwargs'])
    self.assertEqual({'username': 'user'}, config_dict['test_prefix.arguments.sentinel_kwargs'])