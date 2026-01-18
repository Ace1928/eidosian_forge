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
def test_create_server_with_files_fails(self):
    ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, files={'/etc/passwd': 'some data', '/tmp/foo.txt': io.StringIO('data')}, nics='auto')
    self.assertIn('files', str(ex))