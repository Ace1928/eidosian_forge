import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('time.sleep')
@mock.patch('time.time', side_effect=[0, 0.1, 0.6])
@mock.patch.object(nvmeof.Portal, 'reconnect_delay', new_callable=mock.PropertyMock, return_value=10)
@mock.patch.object(nvmeof.Portal, 'is_live', new_callable=mock.PropertyMock)
@mock.patch.object(nvmeof.Target, 'find_device')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock, return_value='connecting')
def test__connect_target_portals_connecting(self, mock_state, mock_rescan, mock_cli, mock_set_ctrls, mock_find_dev, mock_is_live, mock_delay, mock_time, mock_sleep):
    """Test connect target when portals reconnect."""
    mock_is_live.side_effect = [False, False, False, False, True]
    target = self.conn_props.targets[0]
    res = self.connector._connect_target(target)
    self.assertEqual(mock_find_dev.return_value, res)
    self.assertEqual(3, mock_state.call_count)
    self.assertEqual(5, mock_is_live.call_count)
    self.assertEqual(3, mock_delay.call_count)
    self.assertEqual(2, mock_sleep.call_count)
    mock_sleep.assert_has_calls(2 * [mock.call(1)])
    mock_rescan.assert_not_called()
    mock_set_ctrls.assert_called_once()
    mock_find_dev.assert_called_once()
    mock_cli.assert_not_called()