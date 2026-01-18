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
def test_create_server_with_description(self):
    self.cs.servers.create(name='My server', description='descr', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', nics=self._get_server_create_default_nics())
    self.assert_called('POST', '/servers')