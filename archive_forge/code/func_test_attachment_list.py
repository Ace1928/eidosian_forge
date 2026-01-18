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
@ddt.data({'cmd': '', 'expected': ''}, {'cmd': '--volume-id 1234', 'expected': '?volume_id=1234'}, {'cmd': '--status error', 'expected': '?status=error'}, {'cmd': '--all-tenants 1', 'expected': '?all_tenants=1'}, {'cmd': '--all-tenants 1 --volume-id 12345', 'expected': '?all_tenants=1&volume_id=12345'}, {'cmd': '--all-tenants 1 --tenant 12345', 'expected': '?all_tenants=1&project_id=12345'}, {'cmd': '--tenant 12345', 'expected': '?all_tenants=1&project_id=12345'})
@ddt.unpack
def test_attachment_list(self, cmd, expected):
    command = '--os-volume-api-version 3.27 attachment-list '
    command += cmd
    self.run_command(command)
    self.assert_called('GET', '/attachments%s' % expected)