from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_get_server_volume(self):
    v = self.cs.volumes.get_server_volume(1234, 'Work')
    self.assert_request_id(v, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/servers/1234/os-volume_attachments/Work')
    self.assertIsInstance(v, volumes.Volume)