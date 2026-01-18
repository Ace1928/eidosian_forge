from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import fibre_channel_s390x
from os_brick.initiator import linuxfc
from os_brick.tests.initiator import test_connector
def test_get_lun_string(self):
    lun = 1
    lunstring = self.connector._get_lun_string(lun)
    self.assertEqual(lunstring, '0x0001000000000000')
    lun = 255
    lunstring = self.connector._get_lun_string(lun)
    self.assertEqual(lunstring, '0x00ff000000000000')
    lun = 257
    lunstring = self.connector._get_lun_string(lun)
    self.assertEqual(lunstring, '0x0101000000000000')
    lun = 1075855370
    lunstring = self.connector._get_lun_string(lun)
    self.assertEqual(lunstring, '0x4020400a00000000')