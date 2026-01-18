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
@mock.patch.object(nvmeof.LOG, 'error')
@mock.patch.object(nvmeof.Target, 'find_device')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test__connect_target_no_portals_connect(self, mock_state, mock_rescan, mock_cli, mock_set_ctrls, mock_find_dev, mock_log, mock_is_live, mock_delay, mock_time):
    """Test connect target when fails to connect to any portal."""
    retries = 3
    mock_state.side_effect = retries * ['connecting', 'connecting', None]
    mock_cli.side_effect = putils.ProcessExecutionError()
    target = self.conn_props.targets[0]
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector._connect_target, target)
    self.assertEqual(retries, mock_log.call_count)
    self.assertEqual(retries * 3, mock_state.call_count)
    mock_state.assert_has_calls(retries * 3 * [mock.call()])
    mock_rescan.assert_not_called()
    mock_set_ctrls.assert_not_called()
    mock_find_dev.assert_not_called()
    self.assertEqual(3, mock_cli.call_count)
    portal = target.portals[-1]
    mock_cli.assert_has_calls(retries * [mock.call(['connect', '-a', portal.address, '-s', portal.port, '-t', portal.transport, '-n', target.nqn, '-Q', '128', '-l', '-1'])])
    self.assertEqual(retries * 2, mock_is_live.call_count)
    self.assertEqual(retries * 2, mock_delay.call_count)