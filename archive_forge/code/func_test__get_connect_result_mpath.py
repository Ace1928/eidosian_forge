import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
def test__get_connect_result_mpath(self):
    props = self.CON_PROPS.copy()
    props['encrypted'] = False
    res = self.connector._get_connect_result(props, 'wwn', ['sda', 'sdb'], 'mpath')
    expected = {'type': 'block', 'scsi_wwn': 'wwn', 'path': '/dev/mpath', 'multipath_id': 'wwn'}
    self.assertDictEqual(expected, res)