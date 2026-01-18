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
def test_key_is_bytestring(self):
    key = b'\xcf\x84o\xcf\x81\xce\xbdo\xcf\x82'
    encoded = cache._sha1_mangle_key(key)
    self.assertIsNotNone(encoded)