import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_backend_multi_get_data(self):
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    random_key = uuidutils.generate_uuid(dashed=False)
    random_key1 = uuidutils.generate_uuid(dashed=False)
    random_key2 = uuidutils.generate_uuid(dashed=False)
    random_key3 = uuidutils.generate_uuid(dashed=False)
    mapping = {random_key1: 'dummyValue1', random_key2: '', random_key3: 'dummyValue3'}
    region.set_multi(mapping)
    keys = [random_key, random_key1, random_key2, random_key3]
    results = region.get_multi(keys)
    self.assertEqual(NO_VALUE, results[0])
    self.assertEqual('dummyValue1', results[1])
    self.assertEqual('', results[2])
    self.assertEqual('dummyValue3', results[3])