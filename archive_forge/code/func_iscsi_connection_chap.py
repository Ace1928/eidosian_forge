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
def iscsi_connection_chap(self, volume, location, iqn, auth_method, auth_username, auth_password, discovery_auth_method, discovery_auth_username, discovery_auth_password):
    return {'driver_volume_type': 'iscsi', 'data': {'auth_method': auth_method, 'auth_username': auth_username, 'auth_password': auth_password, 'discovery_auth_method': discovery_auth_method, 'discovery_auth_username': discovery_auth_username, 'discovery_auth_password': discovery_auth_password, 'target_lun': 1, 'volume_id': volume['id'], 'target_iqn': iqn, 'target_portal': location}}