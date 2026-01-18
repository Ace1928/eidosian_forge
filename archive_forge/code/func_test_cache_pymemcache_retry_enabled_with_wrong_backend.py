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
def test_cache_pymemcache_retry_enabled_with_wrong_backend(self):
    """Validate we build a config without the retry option when retry
        is disabled.
        """
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='oslo_cache.dict', enable_retry_client=True, retry_attempts=2, retry_delay=2)
    self.assertRaises(exception.ConfigurationError, cache._build_cache_config, self.config_fixture.conf)