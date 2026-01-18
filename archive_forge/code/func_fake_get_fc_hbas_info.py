import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
def fake_get_fc_hbas_info(self):
    hbas = self.fake_get_fc_hbas()
    info = [{'port_name': hbas[0]['port_name'].replace('0x', ''), 'node_name': hbas[0]['node_name'].replace('0x', ''), 'host_device': hbas[0]['ClassDevice'], 'device_path': hbas[0]['ClassDevicePath']}]
    return info