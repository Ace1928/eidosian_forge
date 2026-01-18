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
def test_list_security_group(self):
    s = self.cs.servers.get(1234)
    sgs = s.list_security_group()
    self.assert_request_id(sgs, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/servers/1234/os-security-groups')