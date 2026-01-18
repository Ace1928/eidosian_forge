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
def test_delete_server_meta(self):
    ret = self.cs.servers.delete_meta(1234, ['test_key'])
    self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('DELETE', '/servers/1234/metadata/test_key')