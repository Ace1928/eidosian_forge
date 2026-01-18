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
@ddt.data((False, None), (True, None), (False, 'backend1'), (True, 'backend1'), (False, 'default'), (True, 'default'))
@ddt.unpack
def test_group_failover_replication(self, attach_vol, backend):
    attach = '--allow-attached-volume ' if attach_vol else ''
    backend_id = '--secondary-backend-id ' + backend if backend else ''
    cmd = '--os-volume-api-version 3.38 group-failover-replication 1234 ' + attach + backend_id
    self.run_command(cmd)
    expected = {'failover_replication': {'allow_attached_volume': attach_vol, 'secondary_backend_id': backend if backend else None}}
    self.assert_called('POST', '/groups/1234/action', body=expected)