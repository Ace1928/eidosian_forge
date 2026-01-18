from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import storpool as connector
from os_brick.tests.initiator import test_connector
def fake_volume_list(name):
    self.assertEqual(name, self.adb.volumeName(self.fakeProp['volume']))
    return vdata_list.pop()