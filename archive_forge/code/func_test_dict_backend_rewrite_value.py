from dogpile.cache import region as dp_region
from oslo_cache import core
from oslo_cache.tests import test_cache
from oslo_config import fixture as config_fixture
from oslo_utils import fixture as time_fixture
def test_dict_backend_rewrite_value(self):
    self.region.set(KEY, 'value1')
    self.region.set(KEY, 'value2')
    self.assertEqual('value2', self.region.get(KEY))