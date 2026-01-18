import builtins
import errno
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
import os_brick.privileged as privsep_brick
import os_brick.privileged.nvmeof as privsep_nvme
from os_brick.privileged import rootwrap
from os_brick.tests import base
@mock.patch.object(builtins, 'open', side_effect=Exception)
@mock.patch.object(rootwrap, 'custom_execute')
def test_get_system_uuid_dmidecode(self, mock_exec, mock_open):
    uuid = 'dbc6ba60-36ae-4b96-9310-628832bdfc3d'
    mock_exec.return_value = (f' {uuid} ', '')
    res = privsep_nvme.get_system_uuid()
    self.assertEqual(uuid, res)
    mock_open.assert_called_once_with('/sys/class/dmi/id/product_uuid', 'r')
    mock_exec.assert_called_once_with('dmidecode', '-ssystem-uuid')