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
@mock.patch('time.time', side_effect=[0, 1, 20] * 3)
@mock.patch.object(nvmeof.Portal, 'reconnect_delay', new_callable=mock.PropertyMock, return_value=10)
@mock.patch.object(nvmeof.Portal, 'is_live', new_callable=mock.PropertyMock, return_value=False)
@mock.patch.object(nvmeof.Target, 'find_device')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test__connect_target_portals_down(self, mock_state, mock_rescan, mock_cli, mock_set_ctrls, mock_find_dev, mock_is_live, mock_delay, mock_time):
    """Test connect target has all portal connections down."""
    retries = 3
    mock_state.side_effect = retries * 3 * ['connecting']
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector._connect_target, self.conn_props.targets[0])
    self.assertEqual(retries * 3, mock_state.call_count)
    self.assertEqual(retries * 3, mock_is_live.call_count)
    self.assertEqual(retries * 3, mock_delay.call_count)
    mock_state.assert_has_calls(retries * 3 * [mock.call()])
    mock_rescan.assert_not_called()
    mock_set_ctrls.assert_not_called()
    mock_find_dev.assert_not_called()
    mock_cli.assert_not_called()