import sys
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.common.base import Connection, XmlResponse, JsonResponse
from libcloud.common.types import MalformedResponseError
def test_JsonResponse_class_success(self):
    with requests_mock.mock() as m:
        m.register_uri('GET', 'mock://test.com/', text='{"foo": "bar"}')
        response_obj = requests.get('mock://test.com/')
        response = JsonResponse(response=response_obj, connection=self.mock_connection)
    parsed = response.parse_body()
    self.assertEqual(parsed, {'foo': 'bar'})