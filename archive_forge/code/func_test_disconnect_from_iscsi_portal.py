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
def test_disconnect_from_iscsi_portal(self):
    self.connector._disconnect_from_iscsi_portal(self.CON_PROPS)
    expected_prefix = 'iscsiadm -m node -T %s -p %s ' % (self.CON_PROPS['target_iqn'], self.CON_PROPS['target_portal'])
    expected = [expected_prefix + '--op update -n node.startup -v manual', expected_prefix + '--logout', expected_prefix + '--op delete']
    self.assertListEqual(expected, self.cmds)