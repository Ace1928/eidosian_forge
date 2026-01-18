import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_connection_properties_without_failed_over(self):
    """Handle connection properties with 'failed_over' missing"""
    connection_properties = dict(self.fake_connection_properties)
    connection_properties.pop('failed_over')
    self.connector.connect_volume(connection_properties)
    self.get_password_mock.assert_called_once_with(scaleio.CONNECTOR_CONF_PATH, connection_properties['config_group'], False)