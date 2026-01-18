from dogpile.cache import region as dp_region
from oslo_cache import core
from oslo_cache.tests import test_cache
from oslo_config import fixture as config_fixture
from oslo_utils import fixture as time_fixture
def test_dict_backend(self):
    self.assertIs(NO_VALUE, self.region.get(KEY))
    self.region.set(KEY, VALUE)
    self.assertEqual(VALUE, self.region.get(KEY))
    self.region.delete(KEY)
    self.assertIs(NO_VALUE, self.region.get(KEY))