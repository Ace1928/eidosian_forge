from os_brick.initiator.connectors import local
from os_brick.tests.initiator import test_connector
def test_get_volume_paths(self):
    expected = [self.connection_properties['device_path']]
    actual = self.connector.get_volume_paths(self.connection_properties)
    self.assertEqual(expected, actual)