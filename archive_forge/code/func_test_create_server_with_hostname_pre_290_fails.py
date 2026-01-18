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
def test_create_server_with_hostname_pre_290_fails(self):
    self.cs.api_version = api_versions.APIVersion('2.89')
    ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, nics='auto', hostname='new-hostname')
    self.assertIn("'hostname' argument is only allowed since microversion 2.90", str(ex))