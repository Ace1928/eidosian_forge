import sys
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.common.base import Connection, XmlResponse, JsonResponse
from libcloud.common.types import MalformedResponseError
def test_XmlResponse_class(self):
    with requests_mock.mock() as m:
        m.register_uri('GET', 'mock://test.com/2', text='<foo>bar</foo>')
        response_obj = requests.get('mock://test.com/2')
        response = XmlResponse(response=response_obj, connection=self.mock_connection)
    parsed = response.parse_body()
    self.assertEqual(parsed.tag, 'foo')
    self.assertEqual(parsed.text, 'bar')