import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_correct_read_preference(self):
    self.arguments['read_preference'] = 'secondaryPreferred'
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    self.assertEqual('secondaryPreferred', region.backend.api.read_preference)
    random_key = uuidutils.generate_uuid(dashed=False)
    region.set(random_key, 'dummyValue10')
    self.assertEqual(3, region.backend.api.read_preference)