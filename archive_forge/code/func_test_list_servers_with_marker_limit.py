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
def test_list_servers_with_marker_limit(self):
    sl = self.cs.servers.list(marker=1234, limit=2)
    self.assert_request_id(sl, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/servers/detail?limit=2&marker=1234')
    for s in sl:
        self.assertIsInstance(s, servers.Server)