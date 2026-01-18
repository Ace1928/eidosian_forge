import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_backend_multi_set_get_with_blanks_none(self):
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    random_key = uuidutils.generate_uuid(dashed=False)
    random_key1 = uuidutils.generate_uuid(dashed=False)
    random_key2 = uuidutils.generate_uuid(dashed=False)
    random_key3 = uuidutils.generate_uuid(dashed=False)
    random_key4 = uuidutils.generate_uuid(dashed=False)
    mapping = {random_key1: 'dummyValue1', random_key2: None, random_key3: '', random_key4: 'dummyValue4'}
    region.set_multi(mapping)
    self.assertEqual(NO_VALUE, region.get(random_key))
    self.assertEqual('dummyValue1', region.get(random_key1))
    self.assertIsNone(region.get(random_key2))
    self.assertEqual('', region.get(random_key3))
    self.assertEqual('dummyValue4', region.get(random_key4))
    keys = [random_key, random_key1, random_key2, random_key3, random_key4]
    results = region.get_multi(keys)
    self.assertEqual(NO_VALUE, results[0])
    self.assertEqual('dummyValue1', results[1])
    self.assertIsNone(results[2])
    self.assertEqual('', results[3])
    self.assertEqual('dummyValue4', results[4])
    mapping = {random_key1: 'dummyValue5', random_key2: 'dummyValue6'}
    region.set_multi(mapping)
    self.assertEqual(NO_VALUE, region.get(random_key))
    self.assertEqual('dummyValue5', region.get(random_key1))
    self.assertEqual('dummyValue6', region.get(random_key2))
    self.assertEqual('', region.get(random_key3))