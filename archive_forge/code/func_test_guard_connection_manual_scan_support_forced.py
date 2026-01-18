from unittest import mock
from os_brick.initiator import utils
from os_brick.tests import base
@mock.patch('oslo_concurrency.lockutils.lock')
def test_guard_connection_manual_scan_support_forced(self, mock_lock):
    """Guard locks when cinder forces locking."""
    utils.ISCSI_SUPPORTS_MANUAL_SCAN = True
    with utils.guard_connection({'service_uuid': mock.sentinel.uuid, 'shared_targets': None}):
        mock_lock.assert_called_once_with(mock.sentinel.uuid, 'os-brick-', external=True)