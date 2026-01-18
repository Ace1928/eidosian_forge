import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_incorrect_read_preference(self):
    self.arguments['read_preference'] = 'inValidValue'
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    self.assertEqual('inValidValue', region.backend.api.read_preference)
    random_key = uuidutils.generate_uuid(dashed=False)
    self.assertRaises(ValueError, region.set, random_key, 'dummyValue10')