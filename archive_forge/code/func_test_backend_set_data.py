import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_backend_set_data(self):
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    random_key = uuidutils.generate_uuid(dashed=False)
    region.set(random_key, 'dummyValue')
    self.assertEqual('dummyValue', region.get(random_key))