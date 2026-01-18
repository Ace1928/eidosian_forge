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
def test_cache_region_no_error_multiple_config(self):
    cache.configure_cache_region(self.config_fixture.conf, self.region)
    cache.configure_cache_region(self.config_fixture.conf, self.region)