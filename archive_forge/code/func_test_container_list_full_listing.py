from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
from openstackclient.api import object_store_v1 as object_store
from openstackclient.tests.unit import utils
def test_container_list_full_listing(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '?limit=1&format=json', json=[LIST_CONTAINER_RESP[0]], status_code=200)
    self.requests_mock.register_uri('GET', FAKE_URL + '?marker=%s&limit=1&format=json' % LIST_CONTAINER_RESP[0]['name'], json=[LIST_CONTAINER_RESP[1]], status_code=200)
    self.requests_mock.register_uri('GET', FAKE_URL + '?marker=%s&limit=1&format=json' % LIST_CONTAINER_RESP[1]['name'], json=[], status_code=200)
    ret = self.api.container_list(limit=1, full_listing=True)
    self.assertEqual(LIST_CONTAINER_RESP, ret)