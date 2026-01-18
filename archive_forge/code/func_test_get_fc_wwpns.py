import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test_get_fc_wwpns(self):
    self._set_get_fc_hbas()
    wwpns = self.lfc.get_fc_wwpns()
    expected_wwpns = ['50014380242b9750', '50014380242b9752']
    self.assertEqual(expected_wwpns, wwpns)