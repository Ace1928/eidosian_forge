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
@ddt.data(True, False)
@mock.patch.object(nvmeof.Target, 'find_device')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.NVMeOFConnector, '_do_multipath')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test__connect_target_not_found(self, do_multipath, mock_state, mock_rescan, mock_cli, doing_multipath, mock_set_ctrls, mock_find_dev):
    """Test connect target fails to find device after connecting."""
    self.conn_props.targets[0].portals[-1].controller = 'nvme0'
    doing_multipath.return_value = do_multipath
    retries = 3
    mock_state.side_effect = retries * ['connecting', None, 'live']
    mock_find_dev.side_effect = exception.VolumeDeviceNotFound()
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector._connect_target, self.conn_props.targets[0])
    self.assertEqual(retries * 3, mock_state.call_count)
    mock_state.assert_has_calls(retries * 3 * [mock.call()])
    self.assertEqual(retries, mock_rescan.call_count)
    mock_rescan.assert_has_calls(retries * [mock.call('nvme0')])
    self.assertEqual(retries, mock_set_ctrls.call_count)
    mock_set_ctrls.assert_has_calls(retries * [mock.call()])
    self.assertEqual(retries, mock_find_dev.call_count)
    mock_find_dev.assert_has_calls(retries * [mock.call()])
    if do_multipath:
        self.assertEqual(retries, mock_cli.call_count)
        mock_cli.assert_has_calls(retries * [mock.call(['connect', '-a', 'portal2', '-s', 'port2', '-t', 'tcp', '-n', 'nqn_value', '-Q', '128', '-l', '-1'])])
    else:
        mock_cli.assert_not_called()