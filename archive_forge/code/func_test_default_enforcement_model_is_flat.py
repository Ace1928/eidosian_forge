import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_default_enforcement_model_is_flat(self):
    expected = {'description': 'Limit enforcement and validation does not take project hierarchy into consideration.', 'name': 'flat'}
    self.assertEqual(expected, PROVIDERS.unified_limit_api.get_model())