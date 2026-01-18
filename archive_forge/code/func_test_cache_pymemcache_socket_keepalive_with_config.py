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
def test_cache_pymemcache_socket_keepalive_with_config(self):
    """Validate we build a socket keepalive with the right config."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', enable_socket_keepalive=True, socket_keepalive_idle=12, socket_keepalive_interval=38, socket_keepalive_count=42)
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertTrue(self.config_fixture.conf.cache.enable_socket_keepalive)
    self.assertTrue(config_dict['test_prefix.arguments.socket_keepalive'], KeepaliveOpts)
    self.assertEqual(12, config_dict['test_prefix.arguments.socket_keepalive'].idle)
    self.assertEqual(38, config_dict['test_prefix.arguments.socket_keepalive'].intvl)
    self.assertEqual(42, config_dict['test_prefix.arguments.socket_keepalive'].cnt)