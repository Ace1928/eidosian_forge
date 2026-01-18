import uuid
from dogpile.cache import api as dogpile
from dogpile.cache.backends import memory
from oslo_config import fixture as config_fixture
from keystone.common import cache
import keystone.conf
from keystone.tests import unit
def test_multi_methods_when_invalidating_the_region(self):
    mapping = {uuid.uuid4().hex: uuid.uuid4().hex for _ in range(4)}
    keys = list(mapping.keys())
    values = [mapping[k] for k in keys]
    self._assert_has_no_value(self.region0.get_multi(keys))
    self.region0.set_multi(mapping)
    self.assertEqual(values, self.region0.get_multi(keys))
    self.assertEqual(mapping[keys[0]], self.region0.get(keys[0]))
    self.region1.invalidate()
    self._assert_has_no_value(self.region0.get_multi(keys))