from unittest import mock
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient.tests.unit import utils
from ironicclient.v1 import client
def test_client_no_session(self, http_client_mock):
    self.assertRaisesRegex(TypeError, 'session is required', client.Client, 'http://example.com')