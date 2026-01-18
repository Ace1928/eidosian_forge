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
def test_migrate_volume_bool_force(self):
    self.run_command('--os-volume-api-version 3.16 migrate 1234 fakehost --force-host-copy --lock-volume')
    expected = {'os-migrate_volume': {'force_host_copy': True, 'lock_volume': True, 'host': 'fakehost'}}
    self.assert_called('POST', '/volumes/1234/action', body=expected)