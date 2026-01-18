from dogpile.cache import region as dp_region
from oslo_cache import core
from oslo_cache.tests import test_cache
from oslo_config import fixture as config_fixture
from oslo_utils import fixture as time_fixture
def test_dict_backend_zero_expiration_time(self):
    self.region = dp_region.make_region()
    self.region.configure('oslo_cache.dict', arguments={'expiration_time': 0})
    self.region.set(KEY, VALUE)
    self.time_fixture.advance_time_seconds(1)
    self.assertEqual(VALUE, self.region.get(KEY))
    self.assertEqual(1, len(self.region.backend.cache))
    self.region.backend._clear()
    self.assertEqual(VALUE, self.region.get(KEY))
    self.assertEqual(1, len(self.region.backend.cache))