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
def test_group_type_create_private(self):
    expected = {'group_type': {'name': 'test-type-3', 'description': 'test_type-3-desc', 'is_public': False}}
    self.run_command('--os-volume-api-version 3.11 group-type-create test-type-3 --description=test_type-3-desc --is-public=False')
    self.assert_called('POST', '/group_types', body=expected)