import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
@mock.patch('os.makedirs')
def test_mount(self, mock_makedirs):
    self._sentinel_mount()
    mock_makedirs.assert_called_once()
    processutils.execute.assert_has_calls(self._expected_sentinel_mount_calls())