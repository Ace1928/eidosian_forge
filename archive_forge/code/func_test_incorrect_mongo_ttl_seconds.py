import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_incorrect_mongo_ttl_seconds(self):
    self.arguments['mongo_ttl_seconds'] = 'sixty'
    region = dp_region.make_region()
    self.assertRaises(exception.ConfigurationError, region.configure, 'oslo_cache.mongo', arguments=self.arguments)