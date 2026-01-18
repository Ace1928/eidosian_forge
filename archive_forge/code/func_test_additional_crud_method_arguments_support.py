import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_additional_crud_method_arguments_support(self):
    """Additional arguments should works across find/insert/update."""
    self.arguments['wtimeout'] = 30000
    self.arguments['j'] = True
    self.arguments['continue_on_error'] = True
    self.arguments['secondary_acceptable_latency_ms'] = 60
    region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
    api_methargs = region.backend.api.meth_kwargs
    self.assertEqual(30000, api_methargs['wtimeout'])
    self.assertEqual(True, api_methargs['j'])
    self.assertEqual(True, api_methargs['continue_on_error'])
    self.assertEqual(60, api_methargs['secondary_acceptable_latency_ms'])
    random_key = uuidutils.generate_uuid(dashed=False)
    region.set(random_key, 'dummyValue1')
    self.assertEqual('dummyValue1', region.get(random_key))
    region.set(random_key, 'dummyValue2')
    self.assertEqual('dummyValue2', region.get(random_key))
    random_key = uuidutils.generate_uuid(dashed=False)
    region.set(random_key, 'dummyValue3')
    self.assertEqual('dummyValue3', region.get(random_key))