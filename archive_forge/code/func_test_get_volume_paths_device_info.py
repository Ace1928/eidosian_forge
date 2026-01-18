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
def test_get_volume_paths_device_info(self):
    """Device info path has highest priority."""
    dev_path = '/dev/nvme0n1'
    device_info = {'type': 'block', 'path': dev_path}
    conn_props = connection_properties.copy()
    conn_props['device_path'] = 'lower_priority'
    conn_props = nvmeof.NVMeOFConnProps(conn_props)
    res = self.connector.get_volume_paths(conn_props, device_info)
    self.assertEqual([dev_path], res)