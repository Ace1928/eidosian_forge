import os
import tempfile
from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import huawei
from os_brick.tests.initiator import test_connector
def test_connect_volume_nocli(self):
    """Test the fail connect volume case."""
    self.assertRaises(exception.BrickException, self.connector_nocli.connect_volume, self.connection_properties)