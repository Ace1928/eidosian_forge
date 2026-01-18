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
def test_lock_server_pre_273_fails_with_reason(self):
    self.cs.api_version = api_versions.APIVersion('2.72')
    s = self.cs.servers.get(1234)
    e = self.assertRaises(TypeError, s.lock, reason='blah')
    self.assertIn("unexpected keyword argument 'reason'", str(e))