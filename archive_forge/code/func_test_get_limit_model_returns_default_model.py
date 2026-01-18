import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_limit_model_returns_default_model(self):
    response = self.get('/limits/model')
    model = response.result
    expected = {'model': {'name': 'flat', 'description': 'Limit enforcement and validation does not take project hierarchy into consideration.'}}
    self.assertDictEqual(expected, model)