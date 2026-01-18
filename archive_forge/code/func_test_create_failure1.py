from oslo_serialization import jsonutils
from mistralclient.api import base as api_base
from mistralclient.api.v2 import executions
from mistralclient.tests.unit.v2 import base
def test_create_failure1(self):
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=EXEC, status_code=201)
    self.assertRaises(api_base.APIException, self.executions.create, '')