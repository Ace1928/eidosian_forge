from unittest import mock
from os_brick.initiator import utils
from os_brick.tests import base
@mock.patch('oslo_concurrency.lockutils.lock')
def test_guard_connection_manual_scan_unsupported_not_shared(self, mock_lock):
    utils.ISCSI_SUPPORTS_MANUAL_SCAN = False
    with utils.guard_connection({'shared_targets': False}):
        mock_lock.assert_not_called()