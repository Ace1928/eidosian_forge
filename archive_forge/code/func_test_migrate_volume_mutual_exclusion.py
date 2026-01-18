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
@mock.patch('cinderclient.shell.CinderClientArgumentParser.error')
def test_migrate_volume_mutual_exclusion(self, error_mock):
    error_mock.side_effect = SystemExit
    self.assertRaises(SystemExit, self.run_command, '--os-volume-api-version 3.16 migrate 1234 fakehost --cluster fakecluster')
    msg = 'argument --cluster: not allowed with argument <host>'
    error_mock.assert_called_once_with(msg)