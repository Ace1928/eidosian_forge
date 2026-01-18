import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_cache_configuration_values_assertion(self):
    self.arguments['use_replica'] = True
    self.arguments['replicaset_name'] = 'my_replica'
    self.arguments['mongo_ttl_seconds'] = 60
    self.arguments['ssl'] = False
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    self.assertEqual('localhost:27017', region.backend.api.hosts)
    self.assertEqual('ks_cache', region.backend.api.db_name)
    self.assertEqual('cache', region.backend.api.cache_collection)
    self.assertEqual('test_user', region.backend.api.username)
    self.assertEqual('test_password', region.backend.api.password)
    self.assertEqual(True, region.backend.api.use_replica)
    self.assertEqual('my_replica', region.backend.api.replicaset_name)
    self.assertEqual(False, region.backend.api.conn_kwargs['ssl'])
    self.assertEqual(60, region.backend.api.ttl_seconds)