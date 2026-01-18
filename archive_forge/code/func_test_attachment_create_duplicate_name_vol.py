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
@mock.patch.object(volumes.VolumeManager, 'findall')
def test_attachment_create_duplicate_name_vol(self, mock_findall):
    found = [volumes.Volume(self, {'id': '7654', 'name': 'abc'}, loaded=True), volumes.Volume(self, {'id': '9876', 'name': 'abc'}, loaded=True)]
    mock_findall.return_value = found
    self.assertRaises(exceptions.CommandError, self.run_command, '--os-volume-api-version 3.27 attachment-create abc 789')