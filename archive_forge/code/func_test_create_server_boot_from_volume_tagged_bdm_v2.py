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
def test_create_server_boot_from_volume_tagged_bdm_v2(self):
    bdm = [{'volume_size': '1', 'volume_id': '11111111-1111-1111-1111-111111111111', 'delete_on_termination': '0', 'device_name': 'vda', 'tag': 'foo'}]
    s = self.cs.servers.create(name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', block_device_mapping_v2=bdm)
    self.assert_request_id(s, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers')