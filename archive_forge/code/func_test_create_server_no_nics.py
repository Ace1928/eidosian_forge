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
def test_create_server_no_nics(self):
    """Tests that nics are required in microversion 2.37+
        """
    self.assertRaises(ValueError, self.cs.servers.create, name='test', image='d9d8d53c-4b4a-4144-a5e5-b30d9f1fe46a', flavor='1')