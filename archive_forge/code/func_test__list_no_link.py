from unittest import mock
import requests
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import client
from cinderclient.v3 import volumes
def test__list_no_link(self):
    api = mock.Mock()
    api.client.get.return_value = (mock.sentinel.resp, {'resp_keys': [{'name': '1'}]})
    manager = test_utils.FakeManager(api)
    res = manager._list(mock.sentinel.url, 'resp_keys')
    api.client.get.assert_called_once_with(mock.sentinel.url)
    result = [r.name for r in res]
    self.assertListEqual(['1'], result)