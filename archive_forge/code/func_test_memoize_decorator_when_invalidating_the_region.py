import uuid
from dogpile.cache import api as dogpile
from dogpile.cache.backends import memory
from oslo_config import fixture as config_fixture
from keystone.common import cache
import keystone.conf
from keystone.tests import unit
def test_memoize_decorator_when_invalidating_the_region(self):
    memoize = cache.get_memoization_decorator('cache', region=self.region0)

    @memoize
    def func(value):
        return value + uuid.uuid4().hex
    key = uuid.uuid4().hex
    return_value = func(key)
    self.assertEqual(return_value, func(key))
    self.region1.invalidate()
    new_value = func(key)
    self.assertNotEqual(return_value, new_value)