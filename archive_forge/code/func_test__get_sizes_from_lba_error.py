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
@ddt.data([{'ms': 0, 'ds': 6, 'rp': 0}], [{'ms': 0, 'ds': 9, 'rp': 0}, {'ms': 0, 'ds': 9, 'rp': 0}])
def test__get_sizes_from_lba_error(self, lbafs):
    """Incorrect data returned in LBA information."""
    nsze = 6291456
    ns_data = {'nsze': nsze, 'ncap': nsze, 'nuse': nsze, 'lbafs': lbafs}
    res_nsze, res_size = self.connector._get_sizes_from_lba(ns_data)
    self.assertIsNone(res_nsze)
    self.assertIsNone(res_size)