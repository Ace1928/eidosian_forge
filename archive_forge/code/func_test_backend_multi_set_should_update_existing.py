import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_backend_multi_set_should_update_existing(self):
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    random_key = uuidutils.generate_uuid(dashed=False)
    random_key1 = uuidutils.generate_uuid(dashed=False)
    random_key2 = uuidutils.generate_uuid(dashed=False)
    random_key3 = uuidutils.generate_uuid(dashed=False)
    mapping = {random_key1: 'dummyValue1', random_key2: 'dummyValue2', random_key3: 'dummyValue3'}
    region.set_multi(mapping)
    self.assertEqual(NO_VALUE, region.get(random_key))
    self.assertEqual('dummyValue1', region.get(random_key1))
    self.assertEqual('dummyValue2', region.get(random_key2))
    self.assertEqual('dummyValue3', region.get(random_key3))
    mapping = {random_key1: 'dummyValue4', random_key2: 'dummyValue5'}
    region.set_multi(mapping)
    self.assertEqual(NO_VALUE, region.get(random_key))
    self.assertEqual('dummyValue4', region.get(random_key1))
    self.assertEqual('dummyValue5', region.get(random_key2))
    self.assertEqual('dummyValue3', region.get(random_key3))