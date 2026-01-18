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
def test_cache_dictionary_config_builder_fips_mode_unsupported_redis(self):
    """Validate the FIPS mode is not supported."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.redis', tls_enabled=True, enforce_fips_mode=True)
    self.assertRaises(exception.ConfigurationError, cache._build_cache_config, self.config_fixture.conf)