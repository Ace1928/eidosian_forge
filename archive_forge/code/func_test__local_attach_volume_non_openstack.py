from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(rbd.RBDConnector, '_get_rbd_args')
@mock.patch.object(rbd.RBDConnector, 'create_non_openstack_config')
@mock.patch.object(rbd.RBDConnector, '_execute')
def test__local_attach_volume_non_openstack(self, mock_execute, mock_rbd_cfg, mock_args):
    mock_args.return_value = [mock.sentinel.rbd_args]
    connector = rbd.RBDConnector(None, do_local_attach=True)
    res = connector._local_attach_volume(self.connection_properties)
    mock_rbd_cfg.assert_called_once_with(self.connection_properties)
    mock_args.assert_called_once_with(self.connection_properties, mock_rbd_cfg.return_value)
    self.assertEqual(2, mock_execute.call_count)
    mock_execute.assert_has_calls([mock.call('which', 'rbd'), mock.call('rbd', 'map', 'fake_volume', '--pool', 'fake_pool', mock.sentinel.rbd_args, root_helper=connector._root_helper, run_as_root=True)])
    expected = {'path': '/dev/rbd/fake_pool/fake_volume', 'type': 'block', 'conf': mock_rbd_cfg.return_value}
    self.assertEqual(expected, res)