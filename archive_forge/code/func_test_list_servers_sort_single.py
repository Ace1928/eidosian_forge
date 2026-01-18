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
def test_list_servers_sort_single(self):
    sl = self.cs.servers.list(sort_keys=['display_name'], sort_dirs=['asc'])
    self.assert_request_id(sl, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/servers/detail?sort_dir=asc&sort_key=display_name')
    for s in sl:
        self.assertIsInstance(s, servers.Server)