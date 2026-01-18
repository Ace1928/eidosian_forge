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
def iscsi_connection_multipath(self, volume, locations, iqns, luns):
    return {'driver_volume_type': 'iscsi', 'data': {'volume_id': volume['id'], 'target_portals': locations, 'target_iqns': iqns, 'target_luns': luns}}