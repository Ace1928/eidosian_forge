import uuid
from dogpile.cache import api as dogpile
from dogpile.cache.backends import memory
from oslo_config import fixture as config_fixture
from keystone.common import cache
import keystone.conf
from keystone.tests import unit
def test_direct_region_key_invalidation(self):
    """Invalidate by manually clearing the region key's value.

        NOTE(dstanek): I normally don't like tests that repeat application
        logic, but in this case we need to. There are too many ways that
        the tests above can erroneosly pass that we need this sanity check.
        """
    region_key = cache.RegionInvalidationManager(None, self.region0.name)._region_key
    key = uuid.uuid4().hex
    value = uuid.uuid4().hex
    self.assertIsInstance(self.region0.get(key), dogpile.NoValue)
    self.region0.set(key, value)
    self.assertEqual(value, self.region0.get(key))
    cache.CACHE_INVALIDATION_REGION.delete(region_key)
    self.assertIsInstance(self.region0.get(key), dogpile.NoValue)