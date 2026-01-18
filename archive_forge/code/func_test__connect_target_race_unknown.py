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
@ddt.data((70, '', ''), (errno.EALREADY, '', ''), (1, '', 'already connected'), (1, 'already connected', ''))
@ddt.unpack
@mock.patch.object(nvmeof.LOG, 'warning')
@mock.patch.object(nvmeof.LOG, 'error')
@mock.patch('time.sleep')
@mock.patch('time.time', side_effect=[0, 0.1, 0.6])
@mock.patch.object(nvmeof.Portal, 'reconnect_delay', new_callable=mock.PropertyMock, return_value=10)
@mock.patch.object(nvmeof.Portal, 'is_live', new_callable=mock.PropertyMock)
@mock.patch.object(nvmeof.Target, 'find_device')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test__connect_target_race_unknown(self, exit_code, stdout, stderr, mock_state, mock_rescan, mock_cli, mock_set_ctrls, mock_find_dev, mock_is_live, mock_delay, mock_time, mock_sleep, mock_log_err, mock_log_warn):
    """Test connect target when portal is unknown after race."""
    mock_cli.side_effect = putils.ProcessExecutionError(exit_code=exit_code, stdout=stdout, stderr=stderr)
    mock_state.side_effect = ['connecting', 'connecting', None, 'unknown']
    mock_is_live.side_effect = [False, False, False, True]
    target = self.conn_props.targets[0]
    res = self.connector._connect_target(target)
    self.assertEqual(mock_find_dev.return_value, res)
    self.assertEqual(4, mock_state.call_count)
    self.assertEqual(4, mock_is_live.call_count)
    self.assertEqual(2, mock_delay.call_count)
    self.assertEqual(2, mock_sleep.call_count)
    mock_sleep.assert_has_calls(2 * [mock.call(1)])
    mock_rescan.assert_not_called()
    mock_set_ctrls.assert_called_once()
    mock_find_dev.assert_called_once()
    portal = target.portals[-1]
    mock_cli.assert_called_once_with(['connect', '-a', portal.address, '-s', portal.port, '-t', portal.transport, '-n', target.nqn, '-Q', '128', '-l', '-1'])
    self.assertEqual(1, mock_log_err.call_count)
    self.assertEqual(1, mock_log_warn.call_count)