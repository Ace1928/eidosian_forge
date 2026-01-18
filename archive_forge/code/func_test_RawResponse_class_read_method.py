import sys
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.common.base import Connection, XmlResponse, JsonResponse
from libcloud.common.types import MalformedResponseError
def test_RawResponse_class_read_method(self):
    """
        Test that the RawResponse class includes a response
        property which exhibits the same properties and methods
        as httplib.HTTPResponse for backward compat <1.5.0
        """
    TEST_DATA = '1234abcd'
    conn = Connection(host='mock.com', port=80, secure=False)
    conn.connect()
    with requests_mock.Mocker() as m:
        m.register_uri('GET', 'http://mock.com/raw_data', text=TEST_DATA, headers={'test': 'value'})
        response = conn.request('/raw_data', raw=True)
    data = response.response.read()
    self.assertEqual(data, TEST_DATA)
    header_value = response.response.getheader('test')
    self.assertEqual(header_value, 'value')
    headers = response.response.getheaders()
    self.assertEqual(headers, [('test', 'value')])
    self.assertEqual(response.response.status, 200)