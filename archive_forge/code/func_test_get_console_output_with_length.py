import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
def test_get_console_output_with_length(self):
    success = 'foo'
    s = self.cs.servers.get(1234)
    co = s.get_console_output(length=50)
    self.assert_request_id(co, fakes.FAKE_REQUEST_ID_LIST)
    self.assertEqual(success, s.get_console_output(length=50))
    self.assert_called('POST', '/servers/1234/action')
    co = self.cs.servers.get_console_output(s, length=50)
    self.assert_request_id(co, fakes.FAKE_REQUEST_ID_LIST)
    self.assertEqual(success, self.cs.servers.get_console_output(s, length=50))
    self.assert_called('POST', '/servers/1234/action')