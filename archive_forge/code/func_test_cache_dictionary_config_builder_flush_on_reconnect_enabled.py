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
def test_cache_dictionary_config_builder_flush_on_reconnect_enabled(self):
    """Validate we build a sane dogpile.cache dictionary config."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='oslo_cache.dict', memcache_pool_flush_on_reconnect=True)
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertTrue(self.config_fixture.conf.cache.memcache_pool_flush_on_reconnect)
    self.assertTrue(config_dict['test_prefix.arguments.pool_flush_on_reconnect'])