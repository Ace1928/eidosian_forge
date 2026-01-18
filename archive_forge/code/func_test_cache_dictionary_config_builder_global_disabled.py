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
def test_cache_dictionary_config_builder_global_disabled(self):
    """Validate the backend is reset to default if caching is disabled."""
    self.config_fixture.config(group='cache', enabled=False, config_prefix='test_prefix', backend='oslo_cache.dict')
    self.assertFalse(self.config_fixture.conf.cache.enabled)
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertEqual(_opts._DEFAULT_BACKEND, config_dict['test_prefix.backend'])