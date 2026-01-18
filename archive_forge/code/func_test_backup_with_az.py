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
def test_backup_with_az(self):
    self.run_command('--os-volume-api-version 3.51 backup-create --availability-zone AZ2 --name 1234 1234')
    expected = {'backup': {'volume_id': 1234, 'container': None, 'name': '1234', 'description': None, 'incremental': False, 'force': False, 'snapshot_id': None, 'availability_zone': 'AZ2'}}
    self.assert_called('POST', '/backups', body=expected)