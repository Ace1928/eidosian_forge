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
@mock.patch.object(nvmeof.NVMeOFConnector, '_try_disconnect')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
def test__try_disconnect_all_with_failures(self, mock_set_portals, mock_disconnect):
    """Even with failures it should try to disconnect all portals."""
    exc = exception.ExceptionChainer()
    mock_disconnect.side_effect = [Exception, None]
    self.connector._try_disconnect_all(self.conn_props, exc)
    mock_set_portals.assert_called_once_with()
    self.assertEqual(3, mock_disconnect.call_count)
    mock_disconnect.assert_has_calls((mock.call(self.conn_props.targets[0].portals[0]), mock.call(self.conn_props.targets[0].portals[1]), mock.call(self.conn_props.targets[0].portals[2])))
    self.assertTrue(bool(exc))