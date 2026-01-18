import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test_get_fc_wwnns(self):
    self._set_get_fc_hbas()
    wwnns = self.lfc.get_fc_wwnns()
    expected_wwnns = ['50014380242b9751', '50014380242b9753']
    self.assertEqual(expected_wwnns, wwnns)