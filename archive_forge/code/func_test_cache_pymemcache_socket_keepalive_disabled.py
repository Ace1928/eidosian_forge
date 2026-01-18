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
def test_cache_pymemcache_socket_keepalive_disabled(self):
    """Validate we build a dogpile.cache dict config without keepalive."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', socket_keepalive_idle=2, socket_keepalive_interval=2, socket_keepalive_count=2)
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertFalse(self.config_fixture.conf.cache.enable_socket_keepalive)
    self.assertNotIn('test_prefix.arguments.socket_keepalive', config_dict)