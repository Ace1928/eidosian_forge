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
def test_cache_pymemcache_retry_enabled(self):
    """Validate we build a dogpile.cache dict config with retry."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', enable_retry_client=True)
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    opts = ['enable_retry_client', 'retry_attempts', 'retry_delay']
    for el in opts:
        self.assertIn('test_prefix.arguments.{}'.format(el), config_dict)