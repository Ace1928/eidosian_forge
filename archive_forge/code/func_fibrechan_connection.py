import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
def fibrechan_connection(self, volume, location, wwn, lun=1):
    return {'driver_volume_type': 'fibrechan', 'data': {'volume_id': volume['id'], 'target_portal': location, 'target_wwn': wwn, 'target_lun': lun}}