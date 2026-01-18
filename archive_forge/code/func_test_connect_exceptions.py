from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import storpool as connector
from os_brick.tests.initiator import test_connector
def test_connect_exceptions(self):
    """Raise exceptions on missing connection information"""
    fake = self.fakeProp
    for key in fake.keys():
        c = dict(fake)
        del c[key]
        self.assertRaises(exception.BrickException, self.connector.connect_volume, c)
        if key != 'access_mode':
            self.assertRaises(exception.BrickException, self.connector.disconnect_volume, c, None)