from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_create_volume_with_group(self):
    self.run_command('--os-volume-api-version 3.13 create --group-id 5678 --volume-type 4321 1')
    self.assert_called('GET', '/volumes/1234')
    expected = {'volume': {'imageRef': None, 'size': 1, 'availability_zone': None, 'source_volid': None, 'consistencygroup_id': None, 'group_id': '5678', 'name': None, 'snapshot_id': None, 'metadata': {}, 'volume_type': '4321', 'description': None, 'backup_id': None}}
    self.assert_called_anytime('POST', '/volumes', expected)