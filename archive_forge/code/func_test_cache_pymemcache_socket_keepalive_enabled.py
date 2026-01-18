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
def test_cache_pymemcache_socket_keepalive_enabled(self):
    """Validate we build a dogpile.cache dict config with keepalive."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', enable_socket_keepalive=True)
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertTrue(self.config_fixture.conf.cache.enable_socket_keepalive)
    self.assertIsInstance(config_dict['test_prefix.arguments.socket_keepalive'], KeepaliveOpts)