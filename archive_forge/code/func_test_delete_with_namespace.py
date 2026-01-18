import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import actions
from mistralclient.tests.unit.v2 import base
def test_delete_with_namespace(self):
    url = self.TEST_URL + URL_TEMPLATE_NAME % 'action/namespace'
    m = self.requests_mock.delete(url, status_code=204)
    self.actions.delete('action', 'namespace')
    self.assertEqual(1, m.call_count)