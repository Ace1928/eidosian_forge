import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_multiple_region_cache_configuration(self):
    arguments1 = copy.copy(self.arguments)
    arguments1['cache_collection'] = 'cache_region1'
    region1 = dp_region.make_region().configure('oslo_cache.mongo', arguments=arguments1)
    self.assertEqual('localhost:27017', region1.backend.api.hosts)
    self.assertEqual('ks_cache', region1.backend.api.db_name)
    self.assertEqual('cache_region1', region1.backend.api.cache_collection)
    self.assertEqual('test_user', region1.backend.api.username)
    self.assertEqual('test_password', region1.backend.api.password)
    self.assertIsNone(region1.backend.api._data_manipulator)
    random_key1 = uuidutils.generate_uuid(dashed=False)
    region1.set(random_key1, 'dummyValue10')
    self.assertEqual('dummyValue10', region1.get(random_key1))
    self.assertIsInstance(region1.backend.api._data_manipulator, mongo.BaseTransform)
    class_name = '%s.%s' % (MyTransformer.__module__, 'MyTransformer')
    arguments2 = copy.copy(self.arguments)
    arguments2['cache_collection'] = 'cache_region2'
    arguments2['son_manipulator'] = class_name
    region2 = dp_region.make_region().configure('oslo_cache.mongo', arguments=arguments2)
    self.assertEqual('localhost:27017', region2.backend.api.hosts)
    self.assertEqual('ks_cache', region2.backend.api.db_name)
    self.assertEqual('cache_region2', region2.backend.api.cache_collection)
    self.assertIsNone(region2.backend.api._data_manipulator)
    random_key = uuidutils.generate_uuid(dashed=False)
    region2.set(random_key, 'dummyValue20')
    self.assertEqual('dummyValue20', region2.get(random_key))
    self.assertIsInstance(region2.backend.api._data_manipulator, MyTransformer)
    region1.set(random_key1, 'dummyValue22')
    self.assertEqual('dummyValue22', region1.get(random_key1))